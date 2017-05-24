for p in ("ArgParse","JLD","Knet","AutoGrad")
    if Pkg.installed(p) == nothing; Pkg.add(p); end
end

#Pkg.update()
#Pkg.add("AutoGrad")
#Pkg.checkout("AutoGrad")
#Pkg.add("Knet")
#Pkg.checkout("Knet")
#Pkg.build("Knet")

using ArgParse,JLD,Knet,AutoGrad

!isdefined(:VGG) && include("./vgg.jl")
using VGG

function main(args=ARGS)
    s = ArgParseSettings()
    s.exc_handler=ArgParse.debug_handler
    @add_arg_table s begin
        ("--savefile"; help="Save final model to file")
        ("--loadfile"; default= "modelattdeneme2.jld"; help="Save final model to file")
        ("--epochs"; arg_type=Int; default=5; help="Number of epochs for training.")
        ("--hidden"; nargs='+'; arg_type=Int; default=[512]; help="Sizes of one or more LSTM layers.")
        ("--embed"; arg_type=Int; default=512; help="Size of the embedding vector.")
        ("--batchsize"; arg_type=Int; default=50; help="Number of sequences to train on in parallel.")
        ("--optimization"; default="Adam()"; help="Optimization algorithm and parameters.")
        ("--dropout"; arg_type=Float64; default=0.0; help="Dropout probability.")
        ("--gcheck"; arg_type=Int; default=0; help="Check N random gradients.")
        ("--seed"; arg_type=Int; default=-1; help="Random number seed.")
        ("--atype"; default=(gpu()>=0 ? "KnetArray{Float32}" : "Array{Float32}"); help="array type: Array for cpu, KnetArray for gpu")
        ("--generate"; arg_type=Int; default=12; help="If non-zero generate given number of characters.")
    end
    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true)
    println("opts=",[(k,v) for (k,v) in o]...)
    if o[:seed] > 0; setseed(o[:seed]); end
    atype = eval(parse(o[:atype]))

    global text,vocab,images,ns; (text,vocab),images,ns = loaddata()
    if o[:loadfile]==nothing
      global model = model(atype, o[:hidden], length(vocab), o[:embed])
    else
      info("Loading model from $(o[:loadfile])")
      global model = load(o[:loadfile],"model")
    end
    bleu_scorer(vocab,o[:generate])
    global prms  = initparams(model)
    global data = map(t->minibatch(t, o[:batchsize]), text)
    global index = map(t->index_images(t), text)
    epoch = 0
    losses = report_loss(model,images,index,data[1])

    println("epoch: ", epoch, ", ", "loss", ":", losses)

    for epoch=1:o[:epochs]
        @time train(model, images, index, data[1], prms; pdrop=o[:dropout])

        losses = report_loss(model,images,index,data[1])
        println("epoch: ", epoch, ", ", "loss", ":", losses)
        if o[:gcheck] > 0
            gradcheck(rnnlm, model, rand(data); gcheck=o[:gcheck], verbose=true)
        end
    end

    if o[:savefile] != nothing
        save(o[:savefile], "model", model, "vocab", vocab)
    end

    if o[:generate] > 0
        println("########## SAMPLE CAPTION ############")
        for key in keys(images)
          idx = key
          image_input = VGG.main("Flicker8k_Dataset/$idx.jpg")
          image = reshape(image_input,1,196,512)
          generate(vocab,image,o[:generate])
        end
    end

end

function loaddata()
    f = load("flickr8kconv5.jld")
    images = f["features"]
    data = Any[]
    wordcount = Dict()

    open("Flickr_8k.trainImages.txt") do f
      global train_set = Any[]
      for line in eachline(f)
        push!(train_set,line[1:end-5])
      end
    end

    open("Flickr8k.token.txt") do f
        captions = Any[]
        for line in eachline(f)
            c = split(lowercase(line),['#',' ','\n','\t','.'])
            deleteat!(c, findin(c, [""]))
            cap = c
            push!(captions,cap)
            for word in c[4:end]
              if ~haskey(wordcount,word)
                get!(wordcount,word,0)
              end
              wordcount[word] += 1
            end
        end
        words = keys(wordcount)
        freqs = values(wordcount)
        sorted = sort(collect(zip(freqs,words)))

        sent = Any[]
        for i = 1:length(captions)
          cc = captions[i]
          for j = 4:length(cc)
            if wordcount[cc[j]] <= 5
              cc[j] = "<unk>"
            end
          end
          push!(sent,cc)
        end
        global BOS = 1
        global EOS = 2
        global ns = 0
        vocab = Dict()
        get!(vocab,".",EOS); get!(vocab,"<s>",BOS)

        d = Any[]; nw = 0
        for i = 1:length(sent)
            caption = sent[i]
            if haskey(images,caption[1]) == true && findin(train_set,[caption[1]]) != []
              if length(caption[4:end])<=22
                s = Any[]; ns+=1
                push!(s, caption[1])
                for word in caption[4:end]
                    push!(s, get!(vocab, word, 1+length(vocab))); nw+=1
                end
                push!(d, s)
              #else
              #  s = Any[]; ns+=1
              #  push!(s, caption[1])
              #  for word in caption[4:25]
              #      push!(s, get!(vocab, word, 1+length(vocab))); nw+=1
              #  end
              #  push!(d, s)
              end
            end
        end
        push!(data, d)
    end
    global PAD = length(vocab)+1

    return (data, vocab),images,ns
end

function minibatch(data, batchsize)

    data = sort(data, by=length)
    sequence = Any[]
    for i = 1:length(data)
      push!(sequence,data[i][2:end])
    end
    nbuckets = Int32(floor(length(sequence)/batchsize))
    bos = Int32[]

    for i = 1:batchsize
      push!(bos,BOS)
    end
    buckets = Any[]
    for k = 1:nbuckets
        d = sequence[(k-1)*batchsize+1:k*batchsize]
        bucket = Any[]
        push!(bucket,bos)
        for j = 1:length(d[end])+1
          words = Int32[]
          for i = 1:length(d)
            if j-1 == length(d[i])
              push!(words,EOS)
            elseif j-2 >= length(d[i])
              push!(words,PAD)
            elseif length(d[i]) >= j
              push!(words,d[i][j])
            end
          end
          push!(bucket,words)
        end
        push!(buckets,bucket)
    end
    #=
    for i=1:length(buckets)
      println(length(buckets[i]))
    end
    =#
    return buckets
end

function index_images(data)
    data = sort(data, by=length)
    output = Any[]
    for i = 1:length(data)
      push!(output,data[i][1])
    end
    return output
end

function model(atype, hidden, vocab, embed)
    init(d...)=atype(xavier(Float32,d...))
    bias(d...)=atype(zeros(Float32,d...))
    model = Array(Any, 19)
    param = Dict()
    # Embedding matrix
    model[1] = init(embed,vocab+1)
    # LSTM
    H = hidden[end]
    model[2]   = init(4H,H+embed+512)
    model[3] = bias(4H,1)
    model[3][1:H] = 1
    #Decoding
    model[4] = init(vocab,embed)
    model[5] = bias(vocab,1)
    model[6] = init(512,512)
    #att
    model[7] = init(512,hidden[end])
    model[8] = bias(512,1)
    model[9] = init(512,1)
    model[10] = bias(1,1)
    #selector
    model[11] = init(1,hidden[end])
    model[12] = bias(1,1)
    #initial_hidden/memory
    model[13]=init(512,hidden[end])
    model[14]=bias(1,hidden[end])
    model[15]=init(512,hidden[end])
    model[16]=bias(1,hidden[end])
    #decoding
    model[17]=init(embed,hidden[end])
    model[18]=bias(embed,1)
    #ctx2out
    model[19]=init(embed,512)

    for k = 1:19
      get!(param, k, model[k])
    end
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

function lstm(weight,bias,hidden,cell,input)
    gates   = weight * vcat(hidden, input) .+ bias
    h       = size(hidden,1)
    forget  = sigm(gates[1:h,:])
    ingate  = sigm(gates[1+h:2h,:])
    outgate = sigm(gates[1+2h:3h,:])
    change  = tanh(gates[1+3h:4h,:])
    cell    = cell .* forget + ingate .* change
    hidden  = outgate .* tanh(cell)
    return (hidden,cell)
end

function attention(model, sequence, image; pdrop=0.0)
    batch = length(sequence[1])

    mean_image = mean(image,2); mean_image = reshape(mean_image,batch,512)
    h = tanh(mean_image*model[13].+ model[14]); h = h'
    c = tanh(mean_image*model[15].+ model[16]); c = c'
    total = count = 0
    image_input = reshape(image,batch*196,512)
    projected_features = image_input * model[6]
    projected_features = reshape(projected_features,batch,196,512)

    for t = 1:length(sequence)-1

        att = model[7] * h .+ model[8]; att = reshape(att,batch,1,512)
        attention_input = reshape(tanh(projected_features .+ att),batch*196,512)
        attention_output = reshape(attention_input * model[9] .+ model[10],batch,196)

        attention_output = exp(attention_output)
        alpha = attention_output ./ sum(attention_output,2)
        alpha2 = reshape(alpha,batch,196,1)
        ctx = sum((image .* alpha2),2)
        ctx = reshape(ctx,batch,512); ctx = ctx'

        #selector
        beta =  sigm(model[11] * h .+ model[12])
        ctx = ctx .* beta
        input = model[1][:,sequence[t]]

        input = vcat(input,ctx)
        #LSTM
        (h,c) = lstm(model[2],model[3],h,c,input)

        h = dropout(h,pdrop)

        logits = model[17] * h .+ model[18]
        #ctx2out/prev2out
        logits += model[19] * ctx; logits += model[1][:,sequence[t]]
        logp0 = model[4] * tanh(logits) .+ model[5]

        logp1 = logp(logp0,1)
        golds = sequence[t+1]
        index = golds + size(logp1,1)*(0:(length(golds)-1))
        index = index[golds .!= PAD]
        logp2 = logp1[index]
        total += sum(logp2)
        count += length(index)
    end
    return -total/count
end

function report_loss(model, images, index, data)
    ind = 0
    a = 0
    total = 0
    for sequence in data
        batch = length(sequence[1])
        image = zeros(Float32,batch,196,512)
        for i = ind:ind+batch-1
          j = (i+1) % batch
          if j == 0; j = batch; end
          image[j,:,:] = images[index[1][i+1]]
        end
        ind += batch
        image = convert(KnetArray{Float32},image)
        total += attention(model, sequence, image)
        a += 1
    end
    return -total/a
end

attentiongrad = grad(attention)

function train(model, images, index, data, prms; pdrop=0.0)
    ind = 0
    for sequence in data
        batch = length(sequence[1])
        image = zeros(Float32,batch,196,512)
        for i = ind:ind+batch-1
          j = (i+1) % batch
          if j == 0; j = batch; end
          image[j,:,:] = images[index[1][i+1]]
        end
        ind += batch
        image = convert(KnetArray{Float32},image)
        grads = attentiongrad(model, sequence, image; pdrop=pdrop)
        update!(model, grads, prms)
    end
end

function beam_search(vocab,image,nword,f_candidate)
  index_to_word = Array(String, length(vocab))
  for (k,v) in vocab; index_to_word[v] = k; end
  #input = oftype(state[1], ones(15,1))
  beam_width = 1
  word = 1
  flag2 = true
  prev_prob = 1.0

  input = 1
  batch = 1

  image = convert(KnetArray{Float32},image)
  mean_image = mean(image,2); mean_image = reshape(mean_image,batch,512)
  h = tanh(mean_image*model[13].+ model[14]); h = h'
  c = tanh(mean_image*model[15].+ model[16]); c = c'
  total = count = 0
  image_input = reshape(image,batch*196,512)
  projected_features = image_input * model[6]
  projected_features = reshape(projected_features,batch,196,512)

  while word <= nword
    if word == 1
      global new_sequence = zeros(Float32,beam_width^1,word+1)
    else
      global new_sequence = zeros(Float32,beam_width^2,word+1)
    end
    seq_len = 1

    while flag2 && seq_len <= beam_width

      if word != 1
        input = old_sequence[seq_len,end-1]
        prev_prob = old_sequence[seq_len,end]
        prev_cap = reshape(old_sequence[seq_len,1:end-1],1,size(old_sequence)[2]-1)
        for i = 1:beam_width-1
          prev_cap = vcat(prev_cap,reshape(old_sequence[seq_len,1:end-1],1,size(old_sequence)[2]-1))
        end
      end

      att = model[7] * h .+ model[8]; att = reshape(att,batch,1,512)
      attention_input = reshape(tanh(projected_features .+ att),batch*196,512)
      attention_output = reshape(attention_input * model[9] .+ model[10],batch,196)

      attention_output = exp(attention_output)
      alpha = attention_output ./ sum(attention_output,2)
      alpha2 = reshape(alpha,batch,196,1)
      ctx = sum((image .* alpha2),2)
      ctx = reshape(ctx,batch,512); ctx = ctx'

      #selector
      beta =  sigm(model[11] * h .+ model[12])
      ctx = ctx .* beta
      emb = model[1][:,input]

      inp = vcat(emb,ctx)
      #LSTM
      (h,c) = lstm(model[2],model[3],h,c,inp)

      logits = model[17] * h .+ model[18]
      #ctx2out/prev2out
      logits += model[19] * ctx
      emb = reshape(emb,512,1)
      logits += emb
      logp0 = model[4] * tanh(logits) .+ model[5]

      logp1 = logp(logp0,1)
      p= reshape(convert(Array{Float32},exp(logp1)),length(vocab))

      index = sortperm(p,rev=true)[1:beam_width]
      prob = p[index]*prev_prob

      if word == 1
        new_sequence[1:beam_width,:] = hcat(index,prob)
      else
        index = hcat(prev_cap,index)
        new_sequence[beam_width*(seq_len-1)+1:beam_width*(seq_len+1)-beam_width,:] = hcat(index,prob)
      end

      if word == 1
        flag2 = false
      end

      seq_len += 1

    end
    flag2 = true
    word += 1
    ind = sortperm(new_sequence[:,end],rev=true)
    old_sequence = new_sequence[ind[1:beam_width],:]

  end

  i = findmax(new_sequence[:,end])
  sampled_caption = new_sequence[i[2],1:end-1]
  for j = 1:length(sampled_caption)
    print(index_to_word[convert(Int,sampled_caption[j])])
    print(" ")
    sample_word = index_to_word[convert(Int,sampled_caption[j])]
    write(f_candidate,"$sample_word ")
  end
  write(f_candidate,"\n")
end


function bleu_scorer(vocab,nword)
  f = load("flickr8kconv5.jld")
  images = f["features"]
  open("Flickr_8k.devImages.txt") do f
    global dev_set = Any[]
    for line in eachline(f)
      push!(dev_set,line[1:end-5])
    end
  end

  references= []
  for i = 1:5
    i -= 1
    push!(references,open("./ref$i","w"))
  end
  open("Flickr8k.token.txt") do f
    captions = Any[]
    for line in eachline(f)
      c = split(lowercase(line),['#',' ','\n','\t','.'])
      deleteat!(c, findin(c, [""]))
      push!(captions,c)
    end
    for i = 1:length(dev_set)
      for j = 1:length(captions)
        c = captions[j]
        if findin(c,[dev_set[i]]) != []
          for k = 1:length(c[4:end])
            write(references[parse(Int,c[3])+1],c[k+3])
            write(references[parse(Int,c[3])+1]," ")
          end
          write(references[parse(Int,c[3])+1],"\n")
        end
      end
    end
  end
  for i = 1:length(references); close(references[i]);end


  f_candidate = open("flickr8k_candidate","w")
  for i = 1:length(dev_set)
    image_input = images[dev_set[i]]
    image = reshape(image_input,1,196,512)
    generate(vocab,image,nword,f_candidate)
    println()
  end
  close(f_candidate)

  run(pipeline(`perl multi-bleu.perl ./ref`, stdin="flickr8k_candidate"))

end


function generate(vocab,image,nword,f_candidate)
  index_to_word = Array(String, length(vocab))
  for (k,v) in vocab; index_to_word[v] = k; end

  input = 1
  batch = 1

  image = convert(KnetArray{Float32},image)
  mean_image = mean(image,2); mean_image = reshape(mean_image,batch,512)
  h = tanh(mean_image*model[13].+ model[14]); h = h'
  c = tanh(mean_image*model[15].+ model[16]); c = c'
  total = count = 0
  image_input = reshape(image,batch*196,512)
  projected_features = image_input * model[6]
  projected_features = reshape(projected_features,batch,196,512)

  for t = 1:nword

    att = model[7] * h .+ model[8]; att = reshape(att,batch,1,512)
    attention_input = reshape(tanh(projected_features .+ att),batch*196,512)
    attention_output = reshape(attention_input * model[9] .+ model[10],batch,196)

    attention_output = exp(attention_output)
    alpha = attention_output ./ sum(attention_output,2)
    alpha2 = reshape(alpha,batch,196,1)
    ctx = sum((image .* alpha2),2)
    ctx = reshape(ctx,batch,512); ctx = ctx'

    #selector
    beta =  sigm(model[11] * h .+ model[12])
    ctx = ctx .* beta
    emb = model[1][:,input]

    inp = vcat(emb,ctx)
    #LSTM
    (h,c) = lstm(model[2],model[3],h,c,inp)

    logits = model[17] * h .+ model[18]
    #ctx2out/prev2out
    logits += model[19] * ctx
    emb = reshape(emb,512,1)
    logits += emb
    logp0 = model[4] * tanh(logits) .+ model[5]

    logp1 = logp(logp0,1)
    
    index = sample(exp(logp1))
    print(index_to_word[index])
    print(" ")
    sample_word = index_to_word[index]
    write(f_candidate,"$sample_word ")
    input = index
  end
  write(f_candidate,"\n")
end

function sample(p)
    index = findmax(convert(Array{Float32},p))[2]
    return index
end

main()

