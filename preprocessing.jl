import Random
using MLJ: source, coerce, transform, machine, Multiclass, FeatureSelector, OneHotEncoder, Standardizer
using DataFrames: select, nrow

transform_machine(node, input) = transform(machine(node, input), input)

bool_features = map(Symbol, ["outside.sideline", "outside.baseline", "same.side", "server.is.impact.player"])
cat_features = [:hitpoint, Symbol("previous.hitpoint")]
remove_features = [:id, :train, :gender]

function preprocess_data(data, use_cat::Bool, scale::Bool)
    all_features = setdiff(names(data), [:outcome])
    for feat in bool_features
        data[!, feat] = data[!, feat] .== "TRUE"
    end
    data = data[Random.randperm(nrow(data)), :]
    X = select(data, all_features)
    X = coerce(X, Dict((feat, Multiclass) for feat in cat_features))

    Xs = source(X)
    train_features = setdiff(all_features, remove_features, use_cat ? [] : cat_features)
    res = transform_machine(FeatureSelector(train_features), Xs)
    if use_cat
        res = transform_machine(OneHotEncoder(features = cat_features), res)
    end
    if scale
        scale_features = setdiff(train_features, cat_features, bool_features)
        res = transform_machine(Standardizer(features = scale_features), res)
    end

    return res, data.outcome
end