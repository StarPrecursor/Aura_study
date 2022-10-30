import logging
from pathlib import Path

import joblib
import lightgbm as lgb
import yaml

logger = logging.getLogger("artool")


class FRModel:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.input_ready = False
        self.model_ready = False
        self.evals_result = {}

    def set_inputs(self, x_tr, y_tr, x_val=None, y_val=None, x_te=None, y_te=None):
        self.x_tr = x_tr
        self.y_tr = y_tr
        self.x_val = x_val
        self.y_val = y_val
        self.x_te = x_te
        self.y_te = y_te
        self.input_ready = True


class FRModel_LGB(FRModel):
    def __init__(self, config):
        super().__init__(config)

    def train(self):
        if not self.input_ready:
            logger.error("Input not ready")
            return
        lgb_train = lgb.Dataset(
            self.x_tr, self.y_tr, categorical_feature=self.config["categorical_feature"]
        )
        lgb_eval = lgb.Dataset(
            self.x_val,
            self.y_val,
            categorical_feature=self.config["categorical_feature"],
            reference=lgb_train,
        )
        self.model = lgb.train(
            self.config["params"],
            lgb_train,
            valid_sets=[lgb_train, lgb_eval],
            evals_result=self.evals_result,
            **self.config["train_kargs"],
        )
        self.model_ready = True

    def predict(self, x):
        if self.model_ready:
            return self.model.predict(x)
        else:
            logger.error("Model not ready(trained/loaded)")

    def save_model(self, save_dir, name):
        if not self.model_ready:
            logger.error("Model not ready(trained/loaded)")
            return
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        # save config
        config_path = save_dir / f"{name}.config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(self.config, f)
        # save model
        model_path = save_dir / f"{name}.txt"
        self.model.save_model(str(model_path))
        joblib.dump(self.model, save_dir / f"{name}.pkl")
        # save evals_result to yaml
        joblib.dump(self.evals_result, save_dir / f"{name}.evals_result.pkl")

    def load_model(self, save_dir, name):
        # config
        with open(save_dir / f"{name}.config.yaml") as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        # model
        try:
            self.model = joblib.load(save_dir / f"{name}.pkl")
        except:
            model_path = save_dir / f"{name}.txt"
            self.model = lgb.Booster(model_file=str(model_path))
        self.model_ready = True
        # evals_result
        self.evals_result = joblib.load(save_dir / f"{name}.evals_result.pkl")
