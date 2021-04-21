import threading , concurrent.futures , torch, queue, random
import cv2 as cv
from torch.utils.data import Dataset
from glob import glob
#from torchvision.datasets import ImageFolder
#from torchvision.transforms import ToTensor
#import tensorflow as tf

SIZE_MAX_QUEUE = 50 # 180 200

# Function that performs the work in the thread
def thread_func(files):
	for fn in files
		img = cv.imread(file)
		# save in buffer
		# do we have to resize?
		
    return


# Image class 
class StarWarsDataset(Dataset):
    def __init__(self, path_source_d, path_target_d, shuffle, label, n_thread):
        # Read files
        files_source = glob(path_source_d)
        files_target = glob(path_target_d)
        # Shuffle
        if (shuffle):
            random.shuffle(files_source)
            random.shuffle(files_target)
        
        # Prefetching using threads
        queue_source = queue.Queue(maxsize=SIZE_MAX_QUEUE)
        queue_target = queue.Queue(maxsize=SIZE_MAX_QUEUE)

      #Option 1: ?
        threads_s = []
        threads_t = []
        for _ in range(n_thread)
            thread_source = threading.Thread(target=thread_func, args=[files_source], daemon=True)
            # daemon=True -> The significance of this flag is that the entire Python program exits when only daemon threads are left
            thread_source.start()
            threads_s.append(thread_source)
            
            thread_target = threading.Thread(target=thread_func, args=[files_target], daemon=True) 
            thread_target.start()	
            threads_t.append(thread_target)
            # to wait to another thread to finish we need to call .join()
        
      #Option 2 ?
		with concurrent.futures.ThreadPoolExecutor() as executor:
        	#results = [executor.submit(thread_function, path_algo_d) for _ in range(n_thread)]
        	executor.map(thread_function, )
        	
        	for f in concurrent.futures.as_completed(results):
        		print(f.result()) 

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        idx_img = self.imgs[idx]
        idx_lbl = self.labels[idx]

        if self.transformaciones: # aplicacmos la transformacion (normalizar)
            idx_img = self.transformaciones(idx_img)

            # Dictionary
            muestra = {"imagen" : idx_img,
                      "etiqueta" : idx_lbl}
        return muestra


# Create an instance of our new class
cartoon2D_to_3D = StarWarsDataset(path_source_d, path_target_d, shuffle, label, n_thread)

"""
The *torch.utils.data.DataLoader* class is the main class for loading data. 
This class is given a *Dataset* object as argument.

To use it we have to create an instance of the *DataLoader* class to which we pass the dataset object we have created. 
We define a batch size of 10 and shuffle=False so that the order of the data is not changed at each epoch. 
"""

batch_size=10

train_loader = torch.utils.data.DataLoader(dataset=cartoon2D_to_3D, batch_size=batch_size, shuffle=False)
for i, (numbers, labels) in enumerate(train_loader):
  if  i<11:
    print('Batch number %d'%(i+1))
    print(numbers, labels)



