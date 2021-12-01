'''
    History = model.fit(x=training_set, validation_data=test_set, epochs=EPOCHS)

    plt.plot(History.history['accuracy'])
    plt.plot(History.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('graph/model_accuracy_' + str(models) + '.png')
    plt.show()

    plt.plot(History.history['loss'])
    plt.plot(History.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('graph/model_loss_' + str(models) + '.png')
    plt.show()
    '''

"""
    train(model)
    model = models.load_model('model_best1.h5')

    train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    training_set = train_datagen.flow_from_directory(char_path_train, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
                                                     class_mode='categorical', classes=leaf[0])
    """

'''
    test_generator = test_datagen.flow_from_directory(
        'test',
        target_size=(150, 150),
        batch_size=1,
        class_mode='categorical',
        shuffle=False)



    opt = tf.keras.optimizers.Adam(learning_rate=0.001, amsgrad=True)

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    best_cb = callbacks.ModelCheckpoint('model_best1.h5',
                                        monitor='val_loss',
                                        verbose=1,
                                        save_best_only=True,
                                        save_weights_only=False,
                                        save_freq='epoch',
                                        mode='auto')


    model.fit(
        training_set,
        class_weight=class_weight,
        steps_per_epoch=None,
        epochs=EPOCHS,
        validation_data=test_set,
        #validation_steps=48,
        verbose=1,
        use_multiprocessing=True,
        callbacks=[best_cb])
    '''