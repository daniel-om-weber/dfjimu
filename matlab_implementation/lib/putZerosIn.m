function result = putZerosIn(data,numInpair,numZeros)

l = -numZeros + size(data,1) + (size(data,1)/numInpair)*numZeros;
result  = zeros(l,1);

%make pairs
m = vec2mat(data,numInpair);
datapointer = 1;

    %Place data at ther right spot 
    for t=1:numZeros+numInpair:size(m,1)*(numZeros+numInpair)
        result(t:(t+numInpair-1)) = m(datapointer,:)';
        datapointer = datapointer +1;
    end

end

