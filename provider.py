import threading
import random
import numpy as np
import time

def default_batch_fn(file_data, cur_idx, data_indices, require_size):
    '''
    :param file_data:     [[feats0_0,feats0_1...],[feats1_0,feats1_1,...],...]
    :param cur_idx:
    :param data_indices:
    :param require_size:
    :return: [points_list,labels_list],size
    '''
    end_idx = min(cur_idx + require_size, len(file_data[0]))
    batch_data=[]
    for data_idx in xrange(len(file_data)):
        cur_data=[]
        for idx in data_indices[cur_idx:end_idx]:
            cur_data.append(file_data[data_idx][idx])
        batch_data.append(cur_data)

    return batch_data , end_idx - cur_idx


def default_unpack_feats_labels(batch,num_gpus):
    '''

    :param batch:       [[feats0_0,feats0_1...],[feats1_0,feats1_1,...],...]
    :param num_gpus:
    :return:
    '''
    data_num=len(batch[0])
    if data_num%num_gpus!=0:
        left_num=(data_num/num_gpus+1)*num_gpus-data_num
        left_idx = np.random.randint(0, data_num, left_num)
        for i in xrange(len(batch)):
            for idx in left_idx:
                batch[i].append(batch[i][idx])

    return batch


class Provider(threading.Thread):
    def __init__(self, file_list, model, batch_size, read_fn, batch_fn=default_batch_fn, max_cache=2):
        threading.Thread.__init__(self)

        self.slots = threading.Semaphore(max_cache)
        self.items = threading.Semaphore(0)
        self.mutex = threading.Lock()
        self.thread_end = threading.Event()
        self.data_cache = []

        self.file_list = tuple(file_list)
        self.file_len = len(self.file_list)
        self.indices = range(len(file_list))

        self.file_cur = 0

        self.model = model
        self.read_fn = read_fn
        self.batch_fn = batch_fn

        self.batch_size = batch_size
        self.done = False

        self.start()
        self.cur_data=None

    def run(self):
        model = self.model
        while True:
            for idx in self.indices:
                self.slots.acquire()
                self.mutex.acquire()
                if self.thread_end.is_set():
                    # print 'set'
                    exit(0)

                self.data_cache.append(self.read_fn(model, self.file_list[idx]))

                self.mutex.release()
                self.items.release()

            if self.model == 'train':
                random.shuffle(self.indices)

    def request_data(self):
        '''
        fetch data
        save data in cur_data cur_indices data_index
        :return: success or not
        '''
        file_data=[[]]
        while len(file_data[0])==0:
            if self.file_cur >= self.file_len:
                self.cur_data = None
                return False

            self.items.acquire()
            self.mutex.acquire()
            file_data = self.data_cache.pop(0)
            self.mutex.release()
            self.slots.release()

            self.file_cur += 1

        self.cur_data=file_data
        if self.cur_data is not None:
            self.cur_data_index = 0
            self.cur_indices = range(len(self.cur_data[0]))
            if self.model == 'train':
                random.shuffle(self.cur_indices)
            return True

        return False

    def reset(self):
        self.done=False
        self.file_cur = 0

    def close(self):
        self.thread_end.set()
        self.slots.release()

    def __iter__(self):
        return self

    def next(self):
        if self.done:
            self.reset()
            raise StopIteration

        if self.cur_data is None:
            if not self.request_data():
                raise StopIteration

        batch_data, actual_size = self.batch_fn(self.cur_data, self.cur_data_index, self.cur_indices,self.batch_size)

        self.cur_data_index += actual_size

        left_size = self.batch_size - actual_size

        while self.cur_data_index >= len(self.cur_data[0]):     # reach end file batch, left_size>0 is possible
            if not self.request_data():
                self.done=True
                break

            # data available and we still need to sample
            if left_size > 0:
                left_batch_data, actual_size = self.batch_fn\
                    (self.cur_data, self.cur_data_index, self.cur_indices,left_size)

                for data_idx in xrange(len(batch_data)):
                    batch_data[data_idx]+=left_batch_data[data_idx]

                left_size -= actual_size
                self.cur_data_index += actual_size
            else:
                break

        return batch_data

    def report_status(self):
        print 'cd {} cf {}'.format(self.cur_data_index,self.file_cur)

