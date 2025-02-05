import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def display_random_image(df):
    random_row = df.sample(1).iloc[0]
    filepath, label = random_row['filepaths'], random_row['labels']
    img = Image.open(filepath)
    plt.imshow(img)
    plt.title(f'Label: {label}')
    plt.axis('off')
    plt.show()

def augment_data(train_df, valid_df, test_df, batch_size=16):
    img_size = (256, 256)

    train_datagen = ImageDataGenerator(rotation_range=30, horizontal_flip=True, vertical_flip=True)
    valid_test_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow_from_dataframe(train_df, x_col='filepaths', y_col='labels',
                                                        target_size=img_size, class_mode='categorical', batch_size=batch_size)
    valid_generator = valid_test_datagen.flow_from_dataframe(valid_df, x_col='filepaths', y_col='labels',
                                                             target_size=img_size, class_mode='categorical', batch_size=batch_size)
    test_generator = valid_test_datagen.flow_from_dataframe(test_df, x_col='filepaths', y_col='labels',
                                                            target_size=img_size, class_mode='categorical', batch_size=batch_size)
    return train_generator, valid_generator, test_generator
