# music_tagging
This repository contains code to replicate the results of "Automatic Tagging Using Deep Convolutional Neural Networks” by Choi et al. It takes into account corrections and improvements stated at https://keunwoochoi.wordpress.com/2017/01/12/a-self-critic-on-my-ismir-paper-automatic-tagging-using-deep-convolutional-neural-networks/.

The purpose is to recreate the experiments. If you want pretrained models you can use the models at Keunwoo's own repo: https://github.com/keunwoochoi/music-auto_tagging-keras

The data for Magnatagatune can be found at: http://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset. (Note that you need to concatenate the Audio data files before unzipping them, which can be done with the UNIX command "cat").

Extracted mel features on the audio could take 20 minutes - 1 hour on your machine. Training the model to convergence shouldn't take more than 30 minutes on a modern GPU.

Currently the code creates its own train-validation-test split, but note that to compare performance to previous work you may need to use the "conventional" 13-1-1 split (see here: https://github.com/keunwoochoi/magnatagatune-list).
