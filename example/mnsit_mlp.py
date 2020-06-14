"""
@File : mnsit_mlp.py
@Author: Dong Wang
@Date : 2020/6/14
"""
from SemiFlow.layer import Dense
from SemiFlow.Model import Sequential
from SemiFlow.utils.dataset import mnist

train_set, valid_set, test_set = mnist(one_hot=True)

x_train, y_train = train_set[0], train_set[1]
x_test, y_test = test_set[0], test_set[1]
x_val, y_val = valid_set[0], valid_set[1]

num_classes = 10
batch_size = 64
epochs = 3

model = Sequential()
model.add(Dense(units=256, activation='relu', input_shape=(784,)))
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='RMSprop',
              learning_rate=0.05,
              metrics=['train_loss', 'val_loss'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_val, y_val),
                    # validation_split=0.2,
                    )
print(history.history)
score = model.evaluate(x_test, y_test, verbose=0)
