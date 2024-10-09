"""
prune.py

This modul was designed to prune an already exiting and trained model
It uses the pruning algorithm from keras
First you specify the pruning parameters with which the model_for_pruning is created
Sparsity is a percentage which says how many of the weights are to be set zero
The actual pruning takes place in .fit(), where step for step more percentage of values become zero
In the beginning initial_sparsity % are zero. After each epoch more and more become zero until
final_sparsity is reached. 
Note: You always need more epochs than end_step to reach this goal. end_step is basically how fast
a huge amount of zeros get pruned # But I still dont know EXACTLY how it works

More information can be found here: https://stackoverflow.com/questions/60005900/initial-sparsity-parameter-in-sparsity-polynomialdecay-tensorflow-2-0-magnitud

In the end the pruned model has to be prepared for the export, ie converted back to a "normal" model

Its nothing but a tool to be used by myModel.py which is responsible for creating a finished model
Pruning is only the step after training

Author: Anton Giese
Date: 26.10.2020
"""


import keras
import numpy as np
import tempfile
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import gatherGestures
from  convert import UseAllBits


def prune(model,
          trainX,
          trainY,
          testX,
          testY,
          epochs,
          sparsity):


# --------------------- PRUNE WEIGHTS -------------------- #

      # get the pruning function
      prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

      # Compute end step to finish pruning after the desired amount of epochs.
      batch_size = 128
      validation_split = 0 # 0% of training set will be used for validation set, we already have 
                          # valdiation data

      numGestures = trainX.shape[0] * (1 - validation_split)
      #end_step = np.ceil(numGestures / batch_size).astype(np.int32) * epochs/3
      end_step =6000
      print("Endstep: ",end_step)

      # Define model for pruning.
      pruning_params = {
            'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0,
                                                                  final_sparsity=sparsity,
                                                                  begin_step=0,
                                                                  end_step=end_step)
      }

      model_for_pruning = prune_low_magnitude(model, **pruning_params)

      # `prune_low_magnitude` requires a recompile.
      model_for_pruning.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

      model_for_pruning.summary()

      logdir = tempfile.mkdtemp()

      callbacks = [
      tfmot.sparsity.keras.UpdatePruningStep(),
      tfmot.sparsity.keras.PruningSummaries(log_dir=logdir),
      ]


      model_for_pruning.evaluate(testX,testY)

      # THE ACTUAL PRUNING HAPPENS HERE!!!!!!!!

      model_for_pruning.fit(trainX, trainY, validation_split = validation_split, # = 0.0,
                            epochs = epochs, shuffle = True,callbacks = callbacks,batch_size = batch_size)
                        
      model_for_pruning.evaluate(testX,testY)

      
      # simple calculation for the number of zeros
      total = 0
      nul = 0
      for l in model_for_pruning.layers:
            for rows in l.get_weights()[0]:
                  for v in rows:
                        total += 1
                        if v == 0:
                              nul+= 1
      print("Number total: ",total)
      print("Number null: ",nul)


      # prepare the pruned model for export
      model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
      #model_for_export.save("prunedModel2.h5")

      # return the model
      return model_for_export

# debugging: 
if __name__ == "__main__":
      m = keras.models.load_model("finalModels/basisModelFinal9896.h5")
      g = gatherGestures.gatherGestures()
      x,y,x2,y2 = g.collectAllGestures()
      prune(m,x,y,x2,y2,200,0.7)