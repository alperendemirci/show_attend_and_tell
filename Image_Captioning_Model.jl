for p in ("Knet","ArgParse","AutoGrad","Compat","Images","MAT")
    Pkg.installed(p) == nothing && Pkg.add(p)
end
using Knet,AutoGrad,ArgParse,Compat,Images,MAT

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
      ("--datafiles"; nargs='+'; help="If provided, use first file for training, second for dev, others for test.")
      ("--generate"; arg_type=Int; default=500; help="If non-zero generate given number of characters.")
      ("--hidden";  arg_type=Int; default=200; help="Sizes of one or more LSTM layers.")
      ("--epochs"; arg_type=Int; default=5; help="Number of epochs for training.")
      ("--embed"; arg_type=Int; default=200; help="Size of the embedding vector.")
      ("--batchsize"; arg_type=Int; default=1; help="Number of sequences to train on in parallel.")
      ("--seqlength"; arg_type=Int; default=1; help="Number of steps to unroll the network for.")
      ("--decay"; arg_type=Float64; default=0.9; help="Learning rate decay.")
      ("--lr"; arg_type=Float64; default=1e-1; help="Initial learning rate.")
      ("--gclip"; arg_type=Float64; default=3.0; help="Value to clip the gradient norm at.")
      ("--winit"; arg_type=Float64; default=0.1; help="Initial weights set to winit*randn().")
      ("--gcheck"; arg_type=Int; default=0; help="Check N random gradients.")
      ("--seed"; arg_type=Int; default=38; help="Random number seed.")
      ("--atype"; default=(gpu()>=0 ? "KnetArray{Float32}" : "Array{Float32}"); help="array type: Array for cpu, KnetArray for gpu")
    end
    return parse_args(s;as_symbols = true)
end
!isdefined(:VGG) && include("./vgg.jl")
using VGG

function main(args=ARGS)
    opts = parse_commandline()
    println("opts=",[(k,v) for (k,v) in opts]...)
    opts[:seed] > 0 && srand(opts[:seed])
    opts[:atype] = eval(parse(opts[:atype]))

    caps, vocab = process_flickr30k()
    images = Dict()
    index = Any[]
    for i = 1:length(caps)
      idx = caps[i][1]
      push!(index,idx)
    end
    for i = 1:length(unique(index))
      idx = unique(index)[i]
      features = VGG.main("flickr30k-images/$idx.jpg")
      get!(images,idx,features)
    end


    model = initweights(opts[:atype], opts[:hidden], length(vocab), opts[:winit], opts[:embed])

    prms  = initparams(model)

    state = initstate(opts[:atype],opts[:hidden],opts[:batchsize])

    batch_data = minibatch(caps,vocab,opts[:batchsize],images)
    caps = 0; gc(); Knet.knetgc();
    images = 0; gc(); Knet.knetgc();
    losses = loss(model,copy(state),batch_data)
    println((:epoch,0,:loss,losses...))
    print(sizeof(prms))
    for epoch=1:opts[:epochs]
        @time train(model,prms,copy(state),batch_data;slen = opts[:seqlength],lr = opts[:lr],gclip = opts[:gclip])
        losses = loss(model,copy(state),batch_data)
        println((:epoch,epoch,:loss,losses...))
    end
end

function process_flickr30k()
  wordcount = Dict()
  captions = Any[]
  open("results_20130124.token") do f
    for line in readlines(f)
        caption = split(line,['#',' ','\n','\t','.'])
        deleteat!(caption, findin(caption, [""]))
        cap = caption
        push!(captions,cap)
        for word in caption[4:end]
          if ~haskey(wordcount,word)
            get!(wordcount,word,0)
          end
          wordcount[word] += 1
        end
    end
  end
  words = keys(wordcount)
  freqs = values(wordcount)
  sorted_idx = sort(collect(zip(freqs,words)))
  worddict = Dict()
  for (index,value) in enumerate(sorted_idx)
      get!(worddict, index, value[2])
      index = index
  end
  worddict2 = Dict()
  for (k,v) in worddict
    get!(worddict2,v,length(worddict) + 2 - k)
  end
  #get!(worddict2,"<eos>",1)
  get!(worddict2,"UNK",1)
  indexed_captions = Any[]
  for i =1:length(captions)
    sentence = captions[i][4:end]
    cc =  Any[]
    push!(cc,parse(Int,captions[i][1]))
    for j = 1:length(sentence)
      word_index = worddict2[sentence[j]]
      push!(cc,word_index)
    end
    push!(indexed_captions,cc)
  end
  left_captions = Any[]
  for i=1:length(indexed_captions)
    if 25 > length(indexed_captions[i][2:end])
      push!(left_captions,indexed_captions[i])
    end
  end
  for i = 1:length(left_captions)
    for j = 1:24-length(left_captions[i][2:end])
      push!(left_captions[i],1)
    end
  end

  return left_captions, worddict2
end

function minibatch(indexed_captions,worddict,batchsize,features)
  nbatch = div(length(indexed_captions), batchsize)
  vocab_size = length(worddict)
  data = [ falses(batchsize*24, vocab_size) for i=1:nbatch ]
  image = [ zeros(batchsize,1000) for i=1:nbatch]
  for i = 1:batchsize:nbatch
    for m = i:i+batchsize-1
      for k = 1:24
        data[i][k,indexed_captions[m][k+1]] = 1
      end
      image[i][batchsize,:] = transpose(features[indexed_captions[m][1]])
    end
  end
  return data,image
  #return map(d->convert(KnetArray{Float32},d), data), map(i->convert(KnetArray{Float32},i), image)
end

function initweights(atype, hidden, vocab, winit, embed)
  model = Array(Any, 2*length(hidden)+3)
  param = Dict()
  model[1] = winit*randn(embed+hidden, 4*hidden)
  model[2] = zeros(1, 4*hidden)
  model[2][1:hidden] = 1
  model[end-2] = winit*randn(1000,embed)
  model[end-1] = winit*randn(hidden,vocab)
  model[end] = zeros(1,vocab)
  weight = []
  push!(weight, model[1], model[2], model[end-2], model[end-1], model[end])
  for k = 1:5
    get!(param, k, weight[k])
  end
  # your code ends here
  for k in keys(param); param[k] = convert(atype, param[k]); end
  return param
end

function initparams(model)
    prms = Dict()
    for k in keys(model)
        prms[k] = Adam()
    end
    return prms
end

function initstate(atype,hidden,batchsize)
  state = Array(Any, 2*length(hidden))
  state[1] = zeros(batchsize,hidden)
  state[2] = zeros(batchsize,hidden)
  return map(s->convert(atype,s), state)
end

function lstm(weight,bias,hidden,cell,input)
    gates   = hcat(input,hidden) * weight .+ bias
    hsize   = size(hidden,2)
    forget  = sigm(gates[:,1:hsize])
    ingate  = sigm(gates[:,1+hsize:2hsize])
    outgate = sigm(gates[:,1+2hsize:3hsize])
    change  = tanh(gates[:,1+3hsize:end])
    cell    = cell .* forget + ingate .* change
    hidden  = outgate .* tanh(cell)
    return (hidden,cell)
end

function predict(model,state,input)
  input = input * model[3]
  (state[1],state[2]) = lstm(model[1],model[2],state[1],state[2],input)
  input = state[1]*model[4] .+ model[5]
  return input,state
end

function loss(model,state,data,range=1)
    total = 0.0; count = 0
    atype = typeof(AutoGrad.getval(model[1]))
    for t in range
      input = convert(atype,data[2][t])
      ypred,_ = predict(model,state,input)
      ynorm = logp(ypred,2)
      ygold = data[1][t]
      ygold = convert(atype,ygold)
      total += sum(ygold .* ynorm)
      count += size(ygold,1)
    end
    return -total / count
end

lossgradient = grad(loss);

function train(model, prms, state, data; slen=1, lr=1.0, gclip=0.0)
  for t = 1:slen:length(data[1])-slen

      range = t:t+slen-1

      gloss = lossgradient(model, state, data,range)

      gnorm = 0
      for k in keys(model)
          gnorm += sum(gloss[k].^2);
      end
      gnorm = sqrt(gnorm)

      if gnorm >gclip
          for k in keys(model)
            gloss[k] = (gloss[k] * gclip ) / gnorm
          end
      end

      @time update!(model,gloss,prms)
      gloss = 0, Knet.knetgc();gc();
      isa(state,Vector{Any}) || error("State should not be Boxed.")
      for i = 1:length(state)
          state[i] = AutoGrad.getval(state[i])
      end

  end
end

main()

