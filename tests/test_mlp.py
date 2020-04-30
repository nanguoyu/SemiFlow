"""
@File : test_mlp.py
@Author: Dong Wang
@Date : 2020/4/30
"""

from SemiFlow.layer import Dense
from SemiFlow.Model import Sequential

x_train, y_train = None, None
x_test, y_test = None, None

num_classes = 10
batch_size = 128
epochs = 20

model = Sequential()
model.add(Dense(units=512, activation='relu', input_shape=(784,)))
model.add(Dense(units=512, activation='relu', input_shape=(784,)))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='RMSprop')

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(None, None))
score = model.evaluate(x_test, y_test, verbose=0)
