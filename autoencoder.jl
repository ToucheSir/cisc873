import Random
Random.seed!(42)

using LinearAlgebra
using CSV
using MLJ: source, coerce, Multiclass, machine, transform, fit!, FeatureSelector, OneHotEncoder, Standardizer
using MLDataUtils: shuffleobs, eachbatch
using DataFrames
using Base.Iterators: partition

ENV["CUDA_VISIBLE_DEVICES"] = "1"
ENV["CUDA_PATH"] = "/home/branc/anaconda3/envs/hypotension/lib"
using Flux
using Flux: @epochs, mse, throttle, params
using CuArrays

ENV["GKSwstype"] = "nul"
using StatsPlots
pyplot()

@enum Noise gaussian dropout

scale = true
# scale = false
noise_fn = gaussian
# noise_fn = dropout
dropout_threshold = .10
use_cat = false

import MLJBase
mutable struct StackedDenoisingAutoencoder <: MLJBase.Unsupervised
	noise_type::Noise
	noise_threshold::Real
	epochs::Int
	batch_size::Int

	encoder::Chain
	decoder::Chain
	model::Chain

	function StackedDenoisingAutoencoder(sizes;
		activation=identity,
		noise_type=gaussian,
		noise_threshold=0.10,
		batch_size=32,
		epochs=100)

		encoder = Chain([
			Dense(in_size, out_size, activation)
			for (in_size, out_size) in zip(sizes[1:end-1], sizes[2:end])
		]...) |> gpu
		rev_sizes = reverse(sizes)
		decoder = Chain([
			Dense(in_size, out_size, activation)
			for (in_size, out_size) in zip(rev_sizes[1:end-1], rev_sizes[2:end])
		]...) |> gpu

		new(noise_type, noise_threshold, epochs, batch_size, encoder, decoder, Chain(encoder, decoder))
	end
end
function add_noise(model::StackedDenoisingAutoencoder, X)
	if model.noise_type == gaussian
		X .+ (model.noise_threshold * randn(size(X)))
	else
		X .* (rand(eltype(X), size(X)) .> model.noise_threshold)
	end
end
partition_batch(mat, batch_size) = [hcat(vec.(rows)...) for rows in partition(eachrow(mat), batch_size)]
function MLJBase.fit(model::StackedDenoisingAutoencoder, X)
	x = MLJBase.matrix(X)
	x̃ = add_noise(model, x)

	m = model.model
	loss(x, y) = mse(m(x), y)# + norm(params(model))
	opt = ADAM()
	clean_data = gpu.(partition_batch(x, model.batch_size))
	noisy_data = gpu.(partition_batch(x̃, model.batch_size))

	losses = []
	for epoch in 1:model.epochs
		Flux.train!(loss, params(m), zip(noisy_data, clean_data), opt)
		l = loss(clean_data[1], clean_data[1])
		@info "Epoch" epoch loss=l
		push!(losses, Tracker.data(l))
	end
	return losses
end
function MLJBase.transform(model::StackedDenoisingAutoencoder, _, X)
	h = X |> MLJBase.matrix |> transpose |> cpu(model.encoder) |> Flux.data
	X̂ = cpu(model.decoder)(h).data |> transpose
	X̂, h
end

bool_features = map(Symbol, ["outside.sideline", "outside.baseline", "same.side", "server.is.impact.player"])
cat_features = [:hitpoint, Symbol("previous.hitpoint")]
remove_features = [:id, :train, :gender]
transform_machine(node, input) = transform(machine(node, input), input)
function preprocess_data(data)
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
		res = transform_machine(OneHotEncoder(features=cat_features), res)
	end
	if scale
		scale_features = setdiff(train_features, cat_features, bool_features)
		res = transform_machine(Standardizer(features=scale_features), res)
	end

	return res, data.outcome
end

df = CSV.File("mens_train_file.csv") |> DataFrame
processed, outcome = preprocess_data(df)
fit!(processed)
X = convert(Matrix, processed())

batch_size = 32
input_size = size(X, 2)
latent_size = 3
# latent_size = 2

# sizes = [input_size, 24, 12, 6, latent_size]
sizes = [input_size, 18, 12, 6, latent_size]
# sizes = [input_size, 16, 8, latent_size]
# sizes = [input_size, 16, 8, 4, latent_size]
# sizes = [input_size, 12, latent_size]
# sizes = [input_size, latent_size]
activation = scale ? identity : leakyrelu
sdae = machine(StackedDenoisingAutoencoder(sizes), processed)
fit!(sdae)
X̂, h = transform(sdae, processed)()

plot_name(prefix, extension="png") = "$(join([prefix, join(sizes, '-'), scale ? "scaled" : "raw", noise_fn, batch_size], "_")).$extension"

x1 = X[1, :]
x̂1 = X̂[1, :]
str_features = string.(names(processed()))
plt = plot(bar(str_features, x1, orientation=:h), bar(str_features, x̂1, orientation=:h, ymirror=true), layout=(1,2), legend=false, orientation=:h, yticks=:all)
# plt = bar(str_features, x1 - x̂1, orientation=:h, yticks=:all)
savefig(plt, plot_name("bars"))

res_df = DataFrame(i=1:size(X, 1), outcome=outcome)
group_indices = groupby(res_df, :outcome) |> collect
features = [map(g -> h[j, g.i], group_indices) for j in 1:latent_size]

import Combinatorics

function plot_interactive(plt)
	plotlyjs()
	ret = plot(plt, labels=labels, markersize=2)
	pyplot()
	ret
end

labels = permutedims(unique(df.outcome))
plt_fn = latent_size == 2 ? scatter : scatter3d
marker_size = 7
plts = [plt_fn(feat_data..., markersize=marker_size) for feat_data in Combinatorics.permutations(features)]
all_plts = plot(plts..., layout=length(plts), labels=labels, size=(1500, 1000))
savefig(all_plts, plot_name("scatter"))

iterations = 72
anim_plot = @animate for i in 1:iterations
	plt_fn(features..., markersize=marker_size, camera=(i * 360 / iterations, 30))
end
gif(anim_plot, plot_name("anim_scatter", "gif"), fps=15)

# plot_interactive(plts[1])