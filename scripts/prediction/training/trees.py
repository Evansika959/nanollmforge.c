"""Shared independent-target XGBoost fitting with validation-only stopping."""
import xgboost as xgb

def fit_xgb(x, z, train, stop, seed, depth):
    models = []
    for j in range(3):
        model = xgb.XGBRegressor(n_estimators=1500, learning_rate=.03, max_depth=depth,
            min_child_weight=3, reg_lambda=5, subsample=.85, colsample_bytree=.9,
            objective='reg:squarederror', tree_method='hist', n_jobs=4,
            early_stopping_rounds=50, random_state=seed)
        model.fit(x[train], z[train, j], eval_set=[(x[stop], z[stop, j])], verbose=False)
        models.append(model)
    return models
