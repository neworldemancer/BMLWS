res_dict = {}

lr0 = 0.0005
for it in range(1):
  for lr_fact in [0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300]:
    lr = lr0 * lr_fact

    if lr not in res_dict:
      res_dict[lr] = {'acc':[], 'val_acc':[], 'n_ep':[]}

    x = tf.keras.layers.Input(shape=(256,256,3), dtype=tf.float32)

    base_model.trainable = False
    base_out = base_model(x)

    base_out_f = tf.keras.layers.GlobalAveragePooling2D()(base_out)

    h1 = tf.keras.layers.Dense(64, activation='sigmoid')(base_out_f)
    h2 = tf.keras.layers.Dense(2, activation='softmax')(h1)

    model_aug = tf.keras.Model(x, h2)

    model_aug.compile(optimizer=tf.keras.optimizers.Adam(lr) ,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    save_path = 'save/text_augmented_{epoch}.ckpt'

    batch_size=10
    n_itr_per_epoch = len(x_train) // batch_size
    #save_callback = tf.keras.callbacks.ModelCheckpoint(filepath=save_path,
    #                                                  save_weights_only=True,
    #                                                  save_freq=1 * n_itr_per_epoch) # save every 1 epochs
    
    es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', patience=100) # save every 1 epochs
                                                      
    hist = model_aug.fit(x_train, y_train,
                    epochs=500, batch_size=batch_size, 
                    validation_data=(x_valid, y_valid),
                    callbacks=[es_callback])  # save_callback

    fig, axs = plt.subplots(1, 2, figsize=(10,5))
    axs[0].plot(hist.epoch, hist.history['loss'])
    axs[0].plot(hist.epoch, hist.history['val_loss'])
    axs[0].legend(('training loss', 'validation loss'), loc='lower right')
    axs[1].plot(hist.epoch, hist.history['accuracy'])
    axs[1].plot(hist.epoch, hist.history['val_accuracy'])

    axs[1].legend(('training accuracy', 'validation accuracy'), loc='lower right')
    plt.suptitle(f'lr={lr}, it={it}')
    plt.show()
    plt.close()

    va = np.array(hist.history['val_accuracy'])
    best_val_acc = va.max()
    best_val_acc_ep = va.argmax()
    tr_acc = hist.history['accuracy'][best_val_acc_ep]
    res_dict[lr]['val_acc'].append(best_val_acc)
    res_dict[lr]['n_ep'].append(best_val_acc_ep)
    res_dict[lr]['acc'].append(tr_acc)






lrs = []
v_accs = []
accs = []
eps = []

for lr, d in res_dict.items():
  val_acc = res_dict[lr]['val_acc']
  acc = res_dict[lr]['acc']
  
  n_ep = res_dict[lr]['n_ep']
  
  n = len(acc)

  lrs.extend([lr]*n)
  v_accs.extend(list(val_acc))
  accs.extend(list(acc))
  eps.extend(list(n_ep))
  

plt.semilogx(lrs, accs, '^', label='train acc @ best val acc')
plt.semilogx(lrs, v_accs, '*', label='best val acc')

plt.legend()
plt.xlabel('learning rate')
plt.ylabel('accuracy')
plt.ylim(0.3, 1)
plt.show()
plt.close()

plt.semilogx(lrs, eps, '*')

plt.xlabel('learning rate')
plt.ylabel('# epochs till best val acc')
plt.show()
plt.close()

