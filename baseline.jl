
for p in ("Knet","ArgParse","AutoGrad","Compat","JLD","NPZ")
    Pkg.installed(p) == nothing && Pkg.add(p)
end
using Knet,AutoGrad,ArgParse,Compat,JLD,NPZ

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
      #("--datafiles"; nargs='+'; help="If provided, use first file for training, second for dev, others for test.")
      #("--loadfile"; help="Initialize model from file")
      #("--savefile"; help="Save final model to file")
      #("--bestfile"; help="Save best model to file")
      ("--generate"; arg_type=Int; default=0; help="If non-zero generate given number of characters.")
      ("--epochs"; arg_type=Int; default=1; help="Number of epochs for training.")
      ("--hidden"; nargs='+'; arg_type=Int; default=334; help="Sizes of one or more LSTM layers.")
      ("--embed"; arg_type=Int; default=168; help="Size of the embedding vector.")
      ("--batchsize"; arg_type=Int; default=256; help="Number of sequences to train on in parallel.")
      ("--seqlength"; arg_type=Int; default=100; help="Maximum number of steps to unroll the network for bptt. Initial epochs will use the epoch number as bptt length for faster convergence.")
      ("--optimization"; default="Adam()"; help="Optimization algorithm and parameters.")
      ("--winit"; arg_type=Float64; default=0.1; help="Initial weights set to winit*randn().")
      ("--gcheck"; arg_type=Int; default=0; help="Check N random gradients.")
      ("--seed"; arg_type=Int; default=-1; help="Random number seed.")
      ("--atype"; default=(gpu()>=0 ? "KnetArray{Float32}" : "Array{Float32}"); help="array type: Array for cpu, KnetArray for gpu")
      ("--fast"; action=:store_true; help="skip loss printing for faster run")
      ("--dropout"; arg_type=Float64; default=0.0; help="Dropout probability.")
    end
    return parse_args(s;as_symbols = true)
end


function main(args=ARGS)
    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true)
    if !o[:fast]
        println(s.description)
        println("opts=",[(k,v) for (k,v) in o]...)
    end
    o[:seed] > 0 && srand(o[:seed])
    o[:atype] = eval(parse(o[:atype]))

    train, dev, test =  load_dataset()

    vocab = build_dictionary(train[1])

    data = minibatch([train[1], train[2]], 128, 100)

    model = initweights(opts[:atype], opts[:hidden], length(vocab), opts[:winit], opts[:embed])

    state = initstate(opts[:atype],model,length(data[1]))





end

function load_dataset(load_train = true):
    """
    Load captions and image features
    Possible options: f8k, f30k, coco
    """
    # Captions
    train_caps, dev_caps, test_caps = [],[],[]
    if load_train
        open("/Users/xxx/Documents/f8k/f8k_train_caps.txt") do f
            for lines in readlines(f)
                push!(train_caps, strip(lines))
            end
        end
    else
        train_caps = None
    end
    open("/Users/xxx/Documents/f8k/f8k_dev_caps.txt") do f
        for lines in readlines(f)
            push!(dev_caps, strip(lines))
        end
    end
    open("/Users/xxx/Documents/f8k/f8k_test_caps.txt") do f
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
    dev_ims = npzread("/Users/xxx/Documents/f8k/f8k_dev_ims.npy")
    test_ims = npzread("/Users/xxx/Documents/f8k/f8k_test_ims.npy")

    print(size(dev_caps))
    return (train_caps, train_ims), (dev_caps, dev_ims), (test_caps, test_ims)

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
      get!(worddict2,8254 + 2 - k,v)
    end
    return worddict2, wordcount

end

function minibatch(data, batchsize, maxlen)
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
    push!(feats,features[currentIndices[i,:]])
  end

  return caps,feats
end

function initweights(atype, hidden, vocab, winit, embed)
  model = Array(Any, 2*length(hidden)+3)

  model[1] = w*randn(embed+hidden, 4*hidden)
  model[2] = zeros(1, 4*hidden)
  model[2][1:hidden] = 1 # forget gate bias = 1

  model[end-2] = init(vocab,embed)
  model[end-1] = init(hidden,vocab)
  model[end] = bias(1,vocab)
  return map(m->convert(atype,m), model)
  return model
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

end
