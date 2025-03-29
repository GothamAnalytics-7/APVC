import optuna
import keras
from keras import layers
import tensorflow as tf
from cats_and_dogs import *

##### Objetos do notebooks #####
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),  # Espelhamento horizontal
    layers.RandomRotation(0.1),  # Rotação até ±10%
    layers.RandomZoom(0.1),  # Zoom in/out até ±10%
    layers.RandomContrast(0.1),  # Ajuste de contraste
    layers.RandomBrightness(0.1),  # Ajuste de brilho
])

# Callback para guardar os pesos do melhor modelo
best_model_checkpoint = keras.callbacks.ModelCheckpoint(
    filepath="tmp/optuna_tuning.weights.h5",  # Caminho onde os pesos do melhor modelo serão guardados
    save_weights_only=True,  # Apenas os pesos do modelo são guardados (não a arquitetura completa)
    monitor='val_loss',  # Monitoriza a perda (loss) no conjunto de validação
    mode='min',  # O melhor modelo será aquele que tiver a menor val_loss
    save_best_only=True  # Apenas guarda o modelo se for o melhor encontrado até ao momento
)

# Callback para interromper o treino cedo caso a perda não melhore
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',  # Monitoriza a perda no conjunto de validação
    patience=5  # Se a val_loss não melhorar durante 5 epochs consecutivas, o treino é interrompido
)

# Callback para reduzir a taxa de aprendizagem quando a perda estabiliza
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',  # Monitoriza a perda no conjunto de validação
    factor=0.5,  # Reduz a taxa de aprendizagem para 50% do valor atual
    patience=7,  # Aguarda 7 epochs sem melhoria antes de reduzir a taxa de aprendizagem
    min_lr=1e-6  # Define um limite mínimo para a taxa de aprendizagem (não reduz mais do que isto)
)

################################

optuna_storage_path = "optuna_journal_storage.log"
lock_obj = optuna.storages.journal.JournalFileOpenLock(optuna_storage_path)
storage = optuna.storages.JournalStorage(
    optuna.storages.journal.JournalFileBackend(optuna_storage_path, lock_obj=lock_obj)
)

model = keras.Sequential([
        layers.Input(shape=(img_height, img_width, 3)),
])

def create_model(trial: optuna.Trial) -> keras.Sequential:
    params = {
        "use_data_augmentation"  : trial.suggest_categorical("use_data_augmentation", [False, True]),    # Data Augmentation control
        "use_dropout"            : trial.suggest_categorical("use_dropout", [False, True]),              # Dropout control
        "use_batch_normalization": trial.suggest_categorical("use_batch_normalization", [False, True]),  # Batch Normalization control
        "n_layers_CNN"           : trial.suggest_int("n_layers_CNN", 1, 6),                              # Number of CNN layers control 
        "pool_every_n_layers"    : trial.suggest_int("pool_every_n_layers", 1, 3),                       # When to pool control
        "n_layers_hidden"        : trial.suggest_int("n_layers_hidden", 1, 2)                            # Number of dense layers control
    }
    model = keras.Sequential([
        layers.Input(shape=(img_height, img_width, 3)),
    ])
    if params["use_data_augmentation"]:  # If True, add data augmentation layer
        model.add(data_augmentation)
    for i_layer in range(params["n_layers_CNN"]):  # For suggested number of CNN layers
        params[f"CNN_layer{i_layer}"] = {
            "n_filters"  : trial.suggest_int(f"CNN_layer{i_layer}_n_filters", 4, 64, log=True),       # N filters for current CNN layer
            "kernel_size": trial.suggest_int(f"CNN_layer{i_layer}_kernel_size", 2, 6),                  # Kernel size for current layer
            "activation" : trial.suggest_categorical(f"CNN_layer{i_layer}_activation", [None, "relu"])  # Activation for current layer
        }
        
        model.add(
            layers.Conv2D(
                filters=params[f"CNN_layer{i_layer}"]["n_filters"], 
                kernel_size=params[f"CNN_layer{i_layer}"]["kernel_size"], 
                padding="same",
                activation=None
            )
        )
        if params["use_batch_normalization"]:  # If True, add Batch normalization
            model.add(layers.BatchNormalization())
        if params[f"CNN_layer{i_layer}"]["activation"]:  # If not None, add selected activation layer
            model.add(layers.Activation(params[f"CNN_layer{i_layer}"]["activation"]))
        if i_layer % params["pool_every_n_layers"] == 0:  # If current layer number is divisible by pooling control, add pooling layer
            params[f"CNN_layer{i_layer}"]["pool_size"] = trial.suggest_int(f"CNN_layer{i_layer}_pool_size", 1, 3)  # Pool size for current layer
            model.add(layers.MaxPooling2D(params[f"CNN_layer{i_layer}"]["pool_size"]))
    model.add(layers.Flatten())
    for i_layer in range(params["n_layers_hidden"]):  # For suggested number of hidden layers
        params[f"hidden_layer{i_layer}"] = {
            "n_neurons": trial.suggest_int("n_neurons", 2, 256, log=True)  # N neurons for current hidden layer
        }
        model.add(layers.Dense(params[f"hidden_layer{i_layer}"]["n_neurons"], activation=None))
        if params["use_batch_normalization"]:  # If True, add Batch normalization
            model.add(layers.BatchNormalization())
        model.add(layers.Activation("relu"))
    if params["use_dropout"]:  # If True, add Dropout
        params["dropout_rate"] = trial.suggest_float("dropout_rate", 0.1, 0.5)  # Dropout rate for current model
        model.add(layers.Dropout(params["dropout_rate"]))
    model.add(layers.Dense(1, activation="sigmoid"))  # Output layer
    trial.set_user_attr("model_params", params)
    return model

def create_optimizer(trial: optuna.Trial) -> tf.optimizers.Adam:
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-1, log=True)  # Learning Rate control
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
    trial.set_user_attr("learning_rate", learning_rate)
    return optimizer

def objective(trial: optuna.Trial):
    model = create_model(trial)  # Run model construction optimization
    optimizer = create_optimizer(trial)  # Run optimizer construction optimization
    model.compile(
        optimizer=optimizer, 
        loss='binary_crossentropy', 
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    history = model.fit( 
        train, 
        epochs=50, 
        validation_data=validation,
        callbacks=[early_stopping, reduce_lr]
    )
    score = min(history.history["val_loss"])  # Get best validation loss score
    return score

# Create optuna study with defined function and storage
# Dashboard: 
# optuna-dashboard optuna_journal_storage.log 
# ^^^^^ correr no terminal
study = optuna.create_study(
    direction="minimize",
    storage=storage,
    study_name=f"cats_and_dogs_cnn_model",
    load_if_exists=True
)

# Optimize for n_trials, using one thread, timeout of 5 min
study.optimize(objective, n_trials=100, n_jobs=1, timeout=3600)

print("Best hyperparameters:", study.best_params)
print("Best score", study.best_value)

model = create_model(study.best_trial)
best_optimizer = create_optimizer(study.best_trial)
model.compile(
    optimizer=best_optimizer, 
    loss='binary_crossentropy', 
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)
history = model.fit(
        train, 
        epochs=50, 
        validation_data=validation,
        callbacks=[best_model_checkpoint, early_stopping, reduce_lr]
    )

# Rodar no terminal, no memso diretório que optuna_journal_storage.log: optuna-dashboard optuna_journal_storage.log