import os

import numpy as np
import tensorflow as tf

def image_scaling(img):
    """
    Randomly scales the images between 0.5 to 1.5 times the original size.

    Args:
      img: Training image to scale.
    """
    
    scale = tf.random_uniform([1], minval=0.5, maxval=1.5, dtype=tf.float32, seed=None)
    h_new = tf.to_int32(tf.multiply(tf.to_float(tf.shape(img)[0]), scale))
    w_new = tf.to_int32(tf.multiply(tf.to_float(tf.shape(img)[1]), scale))
    new_shape = tf.squeeze(tf.stack([h_new, w_new]), squeeze_dims=[1])
    img = tf.image.resize_images(img, new_shape)
   
    return img

def image_mirroring(img):
    """
    Randomly mirrors the images.

    Args:
      img: Training image to mirror.
    """
    
    distort_left_right_random = tf.random_uniform([1], 0, 1.0, dtype=tf.float32)[0]
    mirror = tf.less(tf.stack([1.0, distort_left_right_random, 1.0]), 0.5)
    mirror = tf.boolean_mask([0, 1, 2], mirror)
    img = tf.reverse(img, mirror)
    return img

def random_crop_and_pad_image_and_labels(image, crop_h, crop_w):
    """
    Randomly crop and pads the input images.

    Args:
      image: Training image to crop/ pad.
    
      crop_h: Height of cropped segment.
      crop_w: Width of cropped segment.
     
    """

    
    #shape = tf.shape(image)

  
    image_shape = tf.shape(image)
    combined_pad = tf.image.pad_to_bounding_box(image, 0, 0, tf.maximum(crop_h, image_shape[0]), tf.maximum(crop_w, image_shape[1]))
    
    last_image_dim = tf.shape(image)[-1]
  
    combined_crop = tf.random_crop(combined_pad, [crop_h,crop_w,3])
    img_crop = combined_crop[:, :, :last_image_dim]

    
    # Set static shape so that tensorflow knows shape at compile time. 
    img_crop.set_shape((crop_h, crop_w, 3))
    
    return img_crop


def image_resize(image, crop_h, crop_w):
    
    last_image_dim = tf.shape(image)[-1]
    
    img_crop = tf.image.resize_images(image,(crop_h, crop_w,))

    # Set static shape so that tensorflow knows shape at compile time. 
    img_crop.set_shape((crop_h, crop_w, 3))

    return img_crop 

def read_labeled_image_list(data_dir, data_list):
    """Reads txt file containing paths to images and ground truth masks.
    
    Args:
      data_dir: path to the directory with images and masks.
      data_list: path to the file with lines of the form '/path/to/image /path/to/mask'.
       
    Returns:
      Two lists with all file names for images and masks, respectively.
    """
    f = open(data_list, 'r')
    images = []
    for line in f:
        try:
            image= line.strip("\n")
        except ValueError: # Adhoc for test.
            image = mask = line.strip("\n")
        images.append(data_dir + image)
    return images

def read_images_from_disk(input_queue, input_size, random_scale, random_mirror,ignore_label, img_mean): # optional pre-processing arguments
    """Read one image and its corresponding mask with optional pre-processing.
    
    Args:
      input_queue: tf queue with paths to the image and its mask.
      input_size: a tuple with (height, width) values.
                  If not given, return images of original size.
      random_scale: whether to randomly scale the images prior
                    to random crop.
      random_mirror: whether to randomly mirror the images prior
                    to random crop.
      
    Returns:
      Two tensors: the decoded image and its mask.
    """
    img_contents = tf.read_file(input_queue[0])
    
    img = tf.image.decode_jpeg(img_contents, channels=3)

    img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=img)
    img = tf.cast(tf.concat(axis=2, values=[img_b, img_g, img_r]), dtype=tf.float32)
    # Extract mean.
    img -= img_mean

    if input_size is not None:
        h, w = input_size

        # Randomly scale the images.
        if random_scale:
            img = image_scaling(img)

        # Randomly mirror the images.
        if random_mirror:
            img = image_mirroring(img)

        # Randomly crops the images.
        img = random_crop_and_pad_image_and_labels(img, h, w)

    return img





class ImageReader2(object):
    '''Generic ImageReader which reads images and corresponding segmentation
       masks from the disk, and enqueues them into a TensorFlow queue.
    '''

    def __init__(self, data_dir, data_list, input_size, 
                 random_scale, random_mirror,ignore_label, img_mean, coord):
        '''Initialise an ImageReader.
        
        Args:
          data_dir: path to the directory with images and masks.
          data_list: path to the file with lines of the form '/path/to/image /path/to/mask'.
          input_size: a tuple with (height, width) values, to which all the images will be resized.
          random_scale: whether to randomly scale the images prior to random crop.
          random_mirror: whether to randomly mirror the images prior to random crop.

          coord: TensorFlow queue coordinator.
        '''
        self.data_dir = data_dir
        self.data_list = data_list
        self.input_size = input_size
        self.coord = coord
        
        self.image_list = read_labeled_image_list(self.data_dir, self.data_list)
        #print(self.image_list)
        self.images = tf.convert_to_tensor(self.image_list, dtype=tf.string)
        
        self.queue = tf.train.slice_input_producer([self.images],
                                                   shuffle=input_size is not None) # not shuffling if it is val
        self.image = read_images_from_disk(self.queue, self.input_size, random_scale, random_mirror,ignore_label, img_mean) 
        
    def dequeue(self, num_elements):
        '''Pack images and labels into a batch.
        
        Args:
          num_elements: the batch size.
          
        Returns:
          Two tensors of size (batch_size, h, w, {3, 1}) for images and masks.'''
        image_batch = tf.train.batch([self.image], num_elements)
        return image_batch
