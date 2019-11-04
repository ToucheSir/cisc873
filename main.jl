
import Random

import CSV
using MLJ
using DataFrames
import Combinatorics

ENV["GKSwstype"] = "nul"
using StatsPlots
pyplot()

function plot_interactive(plt)
	plotlyjs()
	ret = plot(plt, labels=labels, markersize=2)
	pyplot()
	ret
end

ENV["CUDA_VISIBLE_DEVICES"] = "1"
ENV["CUDA_PATH"] = "/home/branc/anaconda3/envs/hypotension/lib"
using Flux
using Flux: @epochs, mse, throttle, params
using CuArrays
include("preprocessing.jl")

import MLJBase

@enum Noise gaussian dropout

partition_batch(mat, batch_size) = [hcat(vec.(rows)...) for rows in Iterators.partition(eachrow(mat), batch_size)]

mutable struct StackedDenoisingAutoencoder <: MLJBase.Unsupervised
	noise_type::Noise
	noise_threshold::Real
	epochs::Int
	batch_size::Int

	encoder::Chain
	decoder::Chain
	model::Chain

	function StackedDenoisingAutoencoder(
		sizes,
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

function MLJBase.fit(model::StackedDenoisingAutoencoder, X)
	x = MLJBase.matrix(X)
	x̃ = add_noise(model, x)

	m = model.model
	loss(x, y) = mse(m(x), y)# + norm(params(model))
	opt = ADAM()
	clean_data = gpu.(partition_batch(x, model.batch_size))
	noisy_data = gpu.(partition_batch(x̃, model.batch_size))

	print(m)

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

# use_cat = [true, false]
use_cat = [true]
# scale = [true, false]
scale = [false]
# noise_type = [gaussian, dropout]
noise_type = [gaussian]
dropout_threshold = .10

df = CSV.File("mens_train_file.csv") |> DataFrame
for (use_cat, scale, noise_type) in Iterators.product(use_cat, scale, noise_type)
    Random.seed!(42)

    processed, outcome = preprocess_data(df, use_cat, scale)
	fit!(processed)
	processed_df = processed()

	for col in names(processed_df)
		vals = processed_df[!, col]
		min_val = minimum(vals)
		max_val = maximum(vals)
		processed_df[!, col] = (vals .- min_val) ./ (max_val - min_val)
	end

    X = convert(Matrix, processed_df)

    input_size = size(X, 2)
    latent_size = 3
    all_sizes = use_cat ? [
        [input_size, latent_size],
        [input_size, 12, latent_size],
        [input_size, 16, 8, latent_size],
        [input_size, 18, 12, 6, latent_size],
        [input_size, 24, 12, 6, latent_size]
    ] : [
        [input_size, latent_size],
        [input_size, 8, latent_size],
        [input_size, 12, latent_size],
        [input_size, 16, 8, latent_size],
        [input_size, 18, 12, 6, latent_size]
    ]

	for sizes in all_sizes
		function plot_name(name, extension="png") 
			dir = "week2/$(join(sizes, '-'))_$(scale ? "scaled" : "raw")_$noise_type"
			!isdir(dir) && mkdir(dir)
			"$dir/$name.$extension"
		end

        # ae = StackedDenoisingAutoencoder(sizes, scale ? identity : leakyrelu, noise_type)
        ae = StackedDenoisingAutoencoder(sizes, sigmoid, noise_type)
        ae_machine = machine(ae, processed_df)
        fit!(ae_machine)
        X̂, h = transform(ae_machine, processed_df)

        x1 = X[1, :]
        x̂1 = X̂[1, :]
        str_features = string.(names(processed_df))
        plt = plot(
            bar(str_features, x1, orientation=:h),
            bar(str_features, x̂1, orientation=:h, ymirror=true),
            layout=(1,2),
            legend=false,
            orientation=:h,
            yticks=:all
        )
        savefig(plt, plot_name("bars"))
        plt = bar(str_features, x1 - x̂1, orientation=:h, yticks=:all)
        savefig(plt, plot_name("diff_bars"))

        res_df = DataFrame(i=1:size(X, 1), outcome=outcome)
        group_indices = groupby(res_df, :outcome) |> collect
        features = [map(g -> h[j, g.i], group_indices) for j in 1:latent_size]

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
    end
end