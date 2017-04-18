for p in ("Knet","ArgParse","AutoGrad","Compat","Images","MAT","JLD")
    Pkg.installed(p) == nothing && Pkg.add(p)
end
using Knet,AutoGrad,ArgParse,Compat,Images,MAT,JLD

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
      ("--saveimages"; default = "flickr8kfeatures.jld"; help="Save conv5 features to file")
      ("--savecaptions"; default = "flickr8kcaptions.jld"; help="")
      ("--savefile"; default= "model1.jld"; help="Save final model to file")
      ("--datafiles"; nargs='+'; help="If provided, use first file for training, second for dev, others for test.")
      ("--generate"; arg_type=Int; default=15; help="If non-zero generate given number of characters.")
      ("--hidden";  arg_type=Int; default=1000; help="Sizes of one or more LSTM layers.")
      ("--epochs"; arg_type=Int; default=20; help="Number of epochs for training.")
      ("--embed"; arg_type=Int; default=512; help="Size of the embedding vector.")
      ("--batchsize"; arg_type=Int; default=10; help="Number of sequences to train on in parallel.")
      ("--seqlength"; arg_type=Int; default=1; help="Number of steps to unroll the network for.")
      ("--decay"; arg_type=Float64; default=0.9; help="Learning rate decay.")
      ("--lr"; arg_type=Float64; default=1e-1; help="Initial learning rate.")
      ("--gclip"; arg_type=Float64; default=3.0; help="Value to clip the gradient norm at.")
      ("--winit"; arg_type=Float64; default=0.1; help="Initial weights set to winit*randn().")
      ("--gcheck"; arg_type=Int; default=0; help="Check N random gradients.")
      ("--seed"; arg_type=Int; default=38; help="Random number seed.")
      ("--atype"; default=(gpu()>=0 ? "KnetArray{Float32}" : "Array{Float32}"); help="array type: Array for cpu, KnetArray for gpu")
      ("--dropout"; arg_type=Float64; default=0.5; help="Dropout probability.")
      ("--selector"; arg_type=Bool; default=true; help="If true select parts of context")
      ("--prev2out"; arg_type=Bool; default=true; help="Feed previous word into logit" )
      ("--ctx2out"; arg_type=Bool; default=true; help="Feed weighted context into logit" )
      ("--alpha_c"; arg_type=Float64; default=0.0; help="Double stochastic regularization coefficient" )
    end
    return parse_args(s;as_symbols = true)
end
#!isdefined(:VGG) && include("./vgg.jl")
#using VGG

function main(args=ARGS)
    opts = parse_commandline()
    println("opts=",[(k,v) for (k,v) in opts]...)
    opts[:seed] > 0 && srand(opts[:seed])
    opts[:atype] = eval(parse(opts[:atype]))

    captions, vocab, maxlen = process_flickr8k()
    f1 = load("flickr8kfeatures.jld")
    images = f1["images"]
    f2 = load("flickr8kcaptions.jld")
    caps = f2["captions"]

    batch_data = minibatch(caps,vocab,opts[:batchsize],images,maxlen)
    #caps = 0; gc(); Knet.knetgc();
    #images = 0; gc(); Knet.knetgc();

    model = initweights(opts[:atype], opts[:hidden], length(vocab), opts[:winit], opts[:embed])

    prms  = initparams(model)

    state = initstate(opts[:atype],opts[:hidden],opts[:batchsize])

    losses = print_loss(model,copy(state),batch_data,opts[:batchsize],opts[:embed],maxlen,images,caps,opts[:dropout],opts[:selector],opts[:ctx2out],opts[:prev2out])
    println((:epoch,0,:loss,losses...))


    for epoch=1:opts[:epochs]
        @time train(model,prms,copy(state),batch_data,opts[:batchsize],opts[:embed],maxlen,images,caps,opts[:dropout],opts[:selector],opts[:ctx2out],opts[:prev2out];slen = opts[:seqlength],lr = opts[:lr],gclip = opts[:gclip])
        losses = print_loss(model,copy(state),batch_data,opts[:batchsize],opts[:embed],maxlen,images,caps,opts[:dropout],opts[:selector],opts[:ctx2out],opts[:prev2out])
        println((:epoch,epoch,:loss,losses...))
        #=
        if opts[:gcheck] > 0
            gradcheck(loss, model, copy(state), batch_data, 1:opts[:seqlength]; gcheck=opts[:gcheck], verbose=true)
        end
        =#
    end

    if opts[:generate] > 0
        println("########## SAMPLE CAPTION ############")
        image_input = VGG.main("Flicker8k_Dataset/72964268_d532bb8ec7.jpg")
        image_input = reshape(image_input,1,196,512)
        state = initstate(opts[:atype],opts[:hidden],1)
        generate(opts[:atype],model, state, vocab, opts[:generate],image_input,opts[:embed],opts[:dropout],opts[:selector],opts[:ctx2out],opts[:prev2out])
    end
    #=
    if opts[:savefile] != nothing
      info("Saving final model to $(opts[:savefile])")
      for i =1:length(model); model[i] = convert(Array{Float32},model[i]); print(typeof(model[i]));end
      save(opts[:savefile], "model", model, "vocab", vocab)
    end
    =#
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
  maxlen = 16
  left_captions = Any[]
  for i=1:length(indexed_captions)
    if maxlen+1 > length(indexed_captions[i][2:end])
      push!(left_captions,indexed_captions[i])
    end
  end
  for i = 1:length(left_captions)
    for j = 1:(maxlen-1)-length(left_captions[i][2:end])
      push!(left_captions[i],1)
    end
  end

  return left_captions, worddict2, maxlen
end

function process_flickr8k()
  wordcount = Dict()
  captions = Any[]
  open("Flickr8k.token.txt") do f
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
    push!(cc,captions[i][1])
    for j = 1:length(sentence)
      word_index = worddict2[sentence[j]]
      push!(cc,word_index)
    end
    push!(indexed_captions,cc)
  end
  maxlen = 16
  left_captions = Any[]
  for i=1:length(indexed_captions)
    if maxlen > length(indexed_captions[i][2:end])
      push!(left_captions,indexed_captions[i])
    end
  end
  for i = 1:length(left_captions)
    for j = 1:(maxlen-1)-length(left_captions[i][2:end])
      push!(left_captions[i],1)
    end
  end
  #=
  image_indexes = Any[]
  open("Flickr_8k.trainImages.txt") do f
    for line in readlines(f)
      line = split(line,['\n','.'])
      deleteat!(line, findin(line, [""]))
      deleteat!(line, findin(line, ["jpg"]))
      push!(image_indexes,line)
    end
  end
  =#
  return left_captions, worddict2, maxlen
end

function minibatch(indexed_captions,worddict,batchsize,features,maxlen)
  nbatch = div(length(indexed_captions), batchsize) * (maxlen-1)
  vocab_size = length(worddict)
  data = [ falses(batchsize, vocab_size) for i=1:nbatch ]
  batch = 1
  for i = 1:batchsize:nbatch
    for m = i:i+batchsize-1
      for k = 1:(maxlen-1)
        data[batch+k-1][m-i+1,indexed_captions[m][k+1]] = 1
      end
    end
    batch += (maxlen-1)
    if batch > nbatch
      break
    end
  end
  return data
end

function initweights(atype, hidden, vocab, winit, embed)
  model = Array(Any, 17)
  param = Dict()
  model[1] = winit*randn(embed, 4*hidden)
  model[2] = winit*randn(hidden, 4*hidden)
  model[3] = zeros(1, 4*hidden)
  #model[2][1:hidden] = 1
  #encoding
  model[4] = winit*randn(vocab,embed)
  #decoding
  model[5] = winit*randn(hidden,embed)
  model[6] = zeros(1,embed)
  model[7] = winit*randn(embed,vocab)
  model[8] = zeros(1,vocab)
  #projection
  model[9] = winit*randn(512,512)
  #attention
  model[10] = winit*randn(hidden,512)
  model[11] = zeros(1,512)
  model[12] = winit*randn(512,1)
  model[13] = zeros(1,1)
  #
  model[14] = winit*randn(512,4*hidden)
  #selector
  model[15] = winit*randn(hidden,1)
  model[16] = zeros(1,1)
  #ctx2out
  model[17] = winit*randn(512,embed)

  for k = 1:17
    get!(param, k, model[k])
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

function project_features(atype, model, input, batchsize,embed)
  input = reshape(input,batchsize*196,512)
  projected_features = input * model[9]
  projected_features = reshape(projected_features,batchsize,196,512)
  return projected_features
end

function attention(atype, model, state, image_input, projected_features, batchsize)
   att = state[1] * model[10] .+ model[11]
   projected_features = convert(Array{Float32},projected_features)
   att = convert(Array{Float32},att)
   attention_input = tanh(projected_features .+ reshape(att,batchsize,1,512))
   attention_input = convert(atype,attention_input)
   attention_output = reshape((reshape(attention_input,batchsize*196,512) * model[12]).+model[13],batchsize,196)

   alpha = exp(attention_output) ./ sum(exp(attention_output),2)

   alpha2 = convert(Array{Float32},reshape(alpha,batchsize,196,1))
   input = convert(Array{Float32},image_input)
   ctx = sum((input .* alpha2),2)
   ctx = reshape(ctx,batchsize,512)
   return alpha, convert(atype,ctx)
end

function lstm(model,hidden,cell,word_embed,context)
    gates   = word_embed*model[1] + hidden*model[2] .+ model[3] + context*model[14]
    hsize   = size(hidden,2)
    forget  = sigm(gates[:,1:hsize])
    ingate  = sigm(gates[:,1+hsize:2hsize])
    outgate = sigm(gates[:,1+2hsize:3hsize])
    change  = tanh(gates[:,1+3hsize:end])
    cell    = cell .* forget + ingate .* change
    hidden  = outgate .* tanh(cell)
    return (hidden,cell)
end

function predict(atype,model,state,image_input,prev_word,batchsize,embed,prob_dropout,selector,ctx2out,prev2out)
  word_embed = prev_word * model[4]
  #Projection
  projected_features = project_features(atype,model, image_input, batchsize,embed)
  #Attention_Layer
  alpha, context = attention(atype,model, state, image_input, projected_features, batchsize)
  if selector
    beta = state[1] * model[15] .+ model[16]
    context = beta .* context
  end
  (state[1],state[2]) = lstm(model,state[1],state[2],word_embed,context)
  #state[1] = state[1] .* (rand!(similar(AutoGrad.getval(state[1]))) .> prob_dropout) * (1/(1-prob_dropout))
  logits = state[1]*model[5] .+ model[6]
  if ctx2out
    logits += context * model[17]
  end
  if prev2out
    logits += word_embed
  end
  #logits = tanh(logits) .* (rand!(similar(AutoGrad.getval(tanh(logits)))) .> prob_dropout) * (1/(1-prob_dropout))
  words = tanh(logits)*model[7] .+ model[8]
  return words,alpha
end

function generate(atype,model,state,vocab,nchar,image_input,embed,prob_dropout,selector,ctx2out,prev2out)
  index_to_char = Array(String, length(vocab))
  for (k,v) in vocab; index_to_char[v] = k; end
  input = oftype(state[1], zeros(1,length(vocab)))
  index = 1
  image_input = convert(atype,image_input)
  for t in 1:nchar
    ypred,_ = predict(atype,model,state,image_input,input,1,embed,prob_dropout,selector,ctx2out,prev2out)
    input[index] = 0
    index = sample(exp(logp(ypred)))
    println(index_to_char[index])
    input[index] = 1
  end
end

function sample(p)
    p = convert(Array,p)
    r = rand()
    for c = 1:length(p)
        r -= p[c]
        r < 0 && return c
    end
end

function loss(model,state,sentence,image,batchsize,embed,range,index,prob_dropout,selector,ctx2out,prev2out)
    total = 0.0; count = 0
    atype = KnetArray{Float32}
    prev_word = convert(atype,sentence[first(range)])
    #alphas = Any[]
    for t in range
      image_input = convert(atype,image)
      ypred,alpha = predict(atype,model,state,image_input,prev_word,batchsize,embed,prob_dropout,selector,ctx2out,prev2out)
      #push!(alphas,alpha)
      ynorm = logp(ypred,2)
      ygold = sentence[t+1]
      ygold = convert(atype,ygold)
      total += sum(ygold .* ynorm)
      count += size(ygold,1)
      prev_word = ygold
    end


    return -total / count
end

lossgradient = grad(loss);

function print_loss(model, state, data, batchsize,embed,maxlen,features,caps,prob_dropout,selector,ctx2out,prev2out)
  sentence = data

  index = 1
  l = 0
  count = 0
  #nbatch_image = div(length(indexed_captions), batchsize)
  #image = [ zeros(batchsize,196,512) for i=1:nbatch_image]

  #=
  for i = 1:batchsize:nbatch_image
    for m = i:i+batchsize-1
      image[i][m-i+1,:,:] = features[indexed_captions[m][1]]
    end
  end
  =#


  #for t = 1:maxlen:length(sentence)-maxlen
  for t = 1:maxlen:160
    #print(index)
    image = zeros(batchsize,196,512)
    for m = index:index+batchsize-1
      image[m-index+1,:,:] = features[caps[m][1]]
    end
    range = t:t+maxlen-1
    l += loss(model, state, sentence,image, batchsize,embed, range,index,prob_dropout,selector,ctx2out,prev2out)
    index += batchsize
    count += 1

  end

  return -l/count
end

function train(model, prms, state, data, batchsize,embed,maxlen,features,caps,prob_dropout,selector,ctx2out,prev2out; slen=1, lr=1.0, gclip=0.0)
  sentence = data
  index = 1
  #for t = 1:maxlen:length(sentence)-maxlen
  for t = 1:maxlen:160
      #print(index)
      image = zeros(batchsize,196,512)
      for m = index:index+batchsize-1
        image[m-index+1,:,:] = features[caps[m][1]]
      end
      range = t:t+maxlen-1
      gloss = lossgradient(model, state, sentence,image, batchsize,embed, range,index,prob_dropout,selector,ctx2out,prev2out)

      index += batchsize
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

      update!(model,gloss,prms)

      isa(state,Vector{Any}) || error("State should not be Boxed.")
      for i = 1:length(state)
          state[i] = AutoGrad.getval(state[i])
      end

  end
end


main()
