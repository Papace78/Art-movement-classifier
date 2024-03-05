First define a sorter
  params = source_dir, trainval_dir, test_dir

  Sort the dataset into trainval and test directories

Second define a paintings
  params = sorter, batch_size, image_size, validation_split

  Generate the generators

Third create a my_model
  params = input_shape , n_classes , optimizer , fine_tune
