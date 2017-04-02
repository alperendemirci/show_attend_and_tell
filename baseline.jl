for p in ("Knet","ArgParse","AutoGrad","Compat","JLD","NPZ","DataStructures")
    Pkg.installed(p) == nothing && Pkg.add(p)
end
using Knet,AutoGrad,ArgParse,Compat,JLD,NPZ,DataStructures

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
      ("--datafiles"; nargs='+'; help="If provided, use first file for training, second for dev, others for test.")
      ("--generate"; arg_type=Int; default=500; help="If non-zero generate given number of characters.")
      ("--hidden";  arg_type=Int; default=4096; help="Sizes of one or more LSTM layers.")
      ("--epochs"; arg_type=Int; default=5; help="Number of epochs for training.")
      ("--embed"; arg_type=Int; default=4096; help="Size of the embedding vector.")
      #("--batchsize"; arg_type=Int; default=100; help="Number of sequences to train on in parallel.")
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


function main(args=ARGS)
    opts = parse_commandline()
    println("opts=",[(k,v) for (k,v) in opts]...)
    opts[:seed] > 0 && srand(opts[:seed])
    opts[:atype] = eval(parse(opts[:atype]))

    data =  load_dataset()

    vocab, count = build_dictionary(data[1])

    batchdata = minibatch([data[1], data[2]], 128, 100, vocab, 10000)


    model = initweights(opts[:atype], opts[:hidden], length(batchdata[1][1]), opts[:winit], opts[:embed])

    #state = initstate(opts[:atype],opts[:hidden],length(batchdata[1]))
    state = initstate(opts[:atype],opts[:hidden],1)

    #losses = map(d->loss(model,copy(state),d), batchdata)
    losses = loss(model,copy(state),batchdata)
    println((:epoch,0,:loss,losses...))

    for epoch=1:opts[:epochs]
        @time train(model,copy(state),batchdata;slen = opts[:seqlength],lr = opts[:lr],gclip = opts[:gclip])
        losses = map(d->loss(model,copy(state),d),batchdata)

        println((:epoch,epoch,:loss,losses...))
    end


end

function load_dataset(load_train = true):
    """
    Load captions and image features
    Possible options: f8k, f30k, coco
    """
#"/Users/xxx/Documents/f8k/f8k_train_caps.txt"
    # Captions
    train_caps, dev_caps, test_caps = [],[],[]

    if load_train
        open("f8k_train_caps.txt") do f
            for lines in readlines(f)
                push!(train_caps, strip(lines))
            end
        end
    else
        train_caps = None
    end

    open("f8k_dev_caps.txt") do f
        for lines in readlines(f)
            push!(dev_caps, strip(lines))
        end
    end

    open("f8k_test_caps.txt") do f
        for lines in readlines(f)
            push!(test_caps, strip(lines))
        end
    end

    # Image features

    if load_train
        train_ims = npzread("/Users/xxx/Documents/f8k/f8k_train_ims.npy")
    else
        train_ims = None
    end
    dev_ims = npzread("f8k_dev_ims.npy")
    test_ims = npzread("/Users/xxx/Documents/f8k/f8k_test_ims.npy")

    #return (train_caps, train_ims), (dev_caps, dev_ims), (test_caps, test_ims)
    return (train_caps, train_ims)
end


function build_dictionary(text):
    """
    Build a dictionary
    text: list of sentences (pre-tokenized)
    """
    wordcount = OrderedDict()
    for cc in text
        words = split(cc)

        for w in words
            if ~haskey(wordcount,w)
                get!(wordcount,w,0)
            end
            wordcount[w] += 1

        end
    end


    words = keys(wordcount)
    freqs = values(wordcount)

    sorted_idx = sort(collect(zip(freqs,words)))


    worddict = OrderedDict()
    for (index,value) in enumerate(sorted_idx)
        get!(worddict, index, value[2])
        index = index
    end
    worddict2 = OrderedDict()
    for (k,v) in worddict
      get!(worddict2,v,length(worddict) + 2 - k)
    end
    get!(worddict2,"UNK",1)
    get!(worddict2,"<eos>",0)
    return worddict2, wordcount

end


function minibatch(data, batchsize, maxlen, worddict, n_words)
  captions = data[1]

  features = data[2]

  lengths = Any[]
  for cc in captions
    lengths = push!(lengths, length(split(cc)))
  end
  unique_lengths = unique(collect(lengths))
  unique_list = Any[]
  for ll in unique_lengths

    if ll <= maxlen
      push!(unique_list, ll)

    end
  end


  indices = Dict()
  counts = Dict()
  for ll in unique_list
    list = Any[]
    countList = Any[]
    boolean = lengths .== ll
    for i = 1:length(boolean)
      if boolean[i] == true
        push!(list, i)

      end
    end
    push!(countList, length(list))
    get!(indices, ll, list)
    get!(counts, ll, countList)
  end


  #reset
  currentCounts = copy(counts)
  unique_list = shuffle(unique_list)
  indicesPos = Dict()
  for ll in unique_list
    get!(indicesPos, ll, 1)
    indices[ll] = shuffle(indices[ll])
  end
  index = 0


  #next
  count = 0
  while true
    index = mod(index+1, length(unique_list))

    if currentCounts[unique_list[index]][1] > 0
      break
    end

    count += 1
    if count >= length(unique_list)
      break
    end


  end


  #=
  if count >= length(unique_list)
    currentCounts = copy(counts)
    unique_list = shuffle(unique_list)
    indicesPos = Dict()
    for ll in unique_list
      get!(indicesPos, ll, 0)
      indices[ll] = shuffle(indices[ll])
    end
    index = 0
  end
  =#

  #get batchsize
  currentBatchSize = min(batchsize,currentCounts[unique_list[index]][1])
  currentPos = indicesPos[unique_list[index]][1]

  #get indices for current batch
  currentIndices = indices[unique_list[index]][currentPos:currentPos+currentBatchSize-1]
  #=
  indicesPos[indices[unique_list[index]][1] += currentBatchSize
  currentCounts[indices[unique_list[index]][1] -= currentBatchSize
  =#

  caps = Any[]
  feats = Any[]

  for i = 1:length(currentIndices)
    push!(caps,captions[currentIndices[i]])
    push!(feats,features[currentIndices[i],:]')
  end

  sequence= Any[]
  feature_list = Any[]
  caption = Any[]
  for (index, cc) in enumerate(caps)
    caption = Any[]
    for w in split(cc)

      if worddict[w] < n_words
        push!(caption, worddict[w])
      else
        push!(caption,1)
      end
    end

    caption = similar(caption,Float32,1,length(caption))
    push!(sequence,caption)
    push!(feature_list,feats[index])
  end

  return sequence, feature_list

end

function initweights(atype, hidden, vocab, winit, embed)
  model = Array(Any, 2*length(hidden)+3)

  model[1] = winit*randn(embed+hidden, 4*hidden)
  model[2] = zeros(1, 4*hidden)
  model[2][1:hidden] = 1

  model[end-2] = winit*randn(vocab,embed)
  model[end-1] = winit*randn(hidden,vocab)
  model[end] = zeros(1,vocab)
  return map(m->convert(atype,m), model)
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
  #input = input * model[3]
  (state[1],state[2]) = lstm(model[1],model[2],state[1],state[2],input)
  input = state[1]*model[4] .+ model[5]
  return input,state
end

function loss(model,state,data,range=1:length(data[1]))
    total = 0.0; count = 0
    atype = typeof(AutoGrad.getval(model[1]))
    for t in range
      input = convert(atype,data[2][t])
      ypred,s = predict(model,state,input)
      ynorm = logp(ypred,2)
      ygold = data[1][t]
      ygold = convert(atype,ygold)
      total += sum(ygold .* ynorm)
      count += size(ygold,1)
    end
    return -total / count
end

lossgradient = grad(loss);

function train(model, state, data; slen=1, lr=1.0, gclip=0.0)
  #for t = 1:slen:length(data)-slen
      #t = 1
      #range = t:t+slen-1
      range = 1:length(data[1])
      gloss = lossgradient(model, state, data)

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

      isa(state,Vector{Any}) || error("State should not be Boxed.")
      for i = 1:length(state)
          state[i] = AutoGrad.getval(state[i])
      end

  #end
end

main()
