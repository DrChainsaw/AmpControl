package ampcontrol.model.training.model.evolve.transfer;

import org.nd4j.linalg.api.iter.NdIndexIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.indexing.PointIndex;
import org.nd4j.linalg.indexing.SpecifiedIndex;

import java.util.*;
import java.util.stream.IntStream;

import static org.nd4j.linalg.ops.transforms.Transforms.abs;

/**
 * Registry for {@link INDArray}s for which one or more elements are to be transferred to another array. Allows for
 * decisions on how and why to transfer an array to be done as separate activities.
 * <br><br>
 * Note: Creates new INDArrays. Make sure the commit method is called from inside a workspace scope to avoid memory leaks.
 *
 * @author Christian Skärby
 */
class TransferRegistry {

    private final Map<INDArray, ArrayEntry> registry = new HashMap<>();
    private final Map<ArrayEntry, Runnable> actions = new LinkedHashMap<>();

    class ArrayEntry {

        private final INDArray array;
        private final String debugName;
        private final Map<Integer, INDArrayIndex> indexMap = new HashMap<>();

        private ArrayEntry(INDArray array, String debugName) {
            this.array = array;
            this.debugName = debugName;
        }

        // Temporary until a better way is found
        long[] shape() {
            return array.shape();
        }

        void put(ArrayEntry anotherEntry) {
            actions.putIfAbsent(this, () -> this.put(anotherEntry.get()));
        }

        void addIndArrayIndex(int dim, INDArrayIndex index) {
            indexMap.put(dim, merge(index, Optional.ofNullable(indexMap.get(dim))));
        }

        // Temporary until a good way to create these exists
        Comparator<Integer> defaultComparatorFactory(Integer dimension) {
            if (dimension < 2) {
                return new Comparator<Integer>() {

                    final int[] tensorDimensions = IntStream.range(0, array.rank())
                            .filter(dim -> dimension != dim)
                            .toArray();

                    @Override
                    public int compare(Integer e1, Integer e2) {
                        if(e1.equals(e2)) {
                            return 0;
                        }

                        return -Double.compare(
                                abs(array.tensorAlongDimension(e1, tensorDimensions)).sumNumber().doubleValue(),
                                abs(array.tensorAlongDimension(e2, tensorDimensions)).sumNumber().doubleValue());
                    }
                };
            } else {
                return new Comparator<Integer>() {

                    final int[] tensorDimensions = IntStream.range(0, array.rank())
                            .filter(dim -> dimension != dim)
                            .toArray();

                    @Override
                    public int compare(Integer e1, Integer e2) {
                        if(e1.equals(e2)){
                            return 0;
                        }
                        if(!(e1 == 0 && e2 == array.size(dimension)-1)
                        || !(e2 == 0 && e1 == array.size(dimension)-1)) {
                            if (e1 == 0) {
                                return 1;
                            }
                            if (e2 == 0) {
                                return -1;
                            }
                            if(e1 == array.size(dimension)-1) {
                                return 1;
                            }
                            if(e2 == array.size(dimension)-1) {
                                return -1;
                            }
                        }
                        return -Double.compare(
                                abs(array.tensorAlongDimension(e1, tensorDimensions)).sumNumber().doubleValue(),
                                abs(array.tensorAlongDimension(e2, tensorDimensions)).sumNumber().doubleValue());
                    }
                };
            }

        }

        private void put(INDArray anotherArray) {
            try {
                // Something tells me I'll need this soon again...
//                System.out.println(debugName + " assign " + Arrays.toString(anotherArray.shape()) + " to " +
//                        Arrays.toString(array.shape()) + " with " + indexMap);
                // Not just (premature) optimization, this also seems to avoid some exceptions, possibly due to dl4j issue #6327
                if (indexMap.isEmpty()) {
                    // Why check this here? Because Nd4j with GPU backend delays OPs like this so error is thrown due to
                    // some subsequent op
                    if(!Arrays.equals(array.shape(), anotherArray.shape())) {
                        throw new IllegalArgumentException("Tried to assign arrays of different shapes for " + debugName
                                + "!\nThis shape: " + Arrays.toString(array.shape()) + " other shape: "
                                + Arrays.toString(anotherArray.shape()));
                    }
                    array.assign(anotherArray);
                    return;
                }

                TransferRegistry.put(asIndArray(), array, anotherArray);
            } catch (ND4JIllegalStateException e) {
                throw new ND4JIllegalStateException("Could not set array " + debugName + "! Tried to put array of shape "
                        + Arrays.toString(anotherArray.shape()) + " into target array of shape "
                        + Arrays.toString(array.shape()) + " at indexes " + Arrays.toString(asIndArray())
                        + " i.e. target shape " + Arrays.toString(array.get(asIndArray()).shape()), e);
            }
        }

        private INDArray get() {
            try {
                // Not just (premature) optimization, this also seems to avoid some exceptions, possibly due to dl4j issue #6327
                if (indexMap.isEmpty()) {
                    return array.dup();
                }
                return addBackSingletonDimensions(array.get(asIndArray()));
            } catch (ND4JIllegalStateException e) {
                throw new ND4JIllegalStateException("Could not get array " + debugName + "! Target array of shape "
                        + Arrays.toString(array.shape()) + ". Wanted indexes " + Arrays.toString(asIndArray()), e);
            }
        }

        // Workaround for dl4j issue #6341
        INDArray addBackSingletonDimensions(INDArray toReshape) {
            final long[] orgShape = array.shape();
            final long[] newShape = toReshape.shape();

            if (orgShape.length == newShape.length) {
                return toReshape;
            }

            long[] actualShape = orgShape.clone();
            final INDArrayIndex[] reshapeInfo = asIndArray();
            for (int i = 0; i < actualShape.length; i++) {
                // Assumes NDArrayIndex.all() returns 0 here
                actualShape[i] = reshapeInfo[i].length() == 0 ? actualShape[i] : reshapeInfo[i].length();
            }
            return toReshape.reshape(actualShape);
        }


        private INDArrayIndex merge(INDArrayIndex index1, Optional<INDArrayIndex> index2) {
            if (!index2.isPresent()) {
                return index1;
            }

            throw new UnsupportedOperationException("Not implemented yet! name: " + debugName + " ind1 " + index1 + " ind2: " + index2.get());
        }

        private INDArrayIndex[] asIndArray() {
            return IntStream.range(0, array.rank())
                    .mapToObj(dim -> indexMap.getOrDefault(dim, NDArrayIndex.all()))
                    .peek(INDArrayIndex::reset)
                    .toArray(INDArrayIndex[]::new);
        }

    }

    ArrayEntry register(INDArray array) {
        return register(array, String.valueOf(registry.size()));
    }

    ArrayEntry register(INDArray array, String debugName) {
        return registry.computeIfAbsent(Objects.requireNonNull(array, "Got null array for " + debugName), arr -> new ArrayEntry(array, debugName));
    }

    void commit() {
        actions.values().forEach(Runnable::run);
        actions.clear();
        registry.clear();
    }

    /**
     * Workaround for nd4j 1.0.0-beta2 not being able to do INDArray#put(INDArrayIndex[] indices, INDArray element)
     * when one of indices is of type SpecifiedIndex. Method is copy pasted from BaseNDArray SNAPSHOT 2018-10-24.
     * @param indices the indices to put the ndarray in to
     * @param target the nrarray to put source to in indices
     * @param source the ndarray to put
     * @return target (after modification)
     */
    public static INDArray put(INDArrayIndex[] indices, INDArray target, INDArray source) {
        Nd4j.getCompressor().autoDecompress(target);
        boolean isSpecifiedIndex = false;
        for(INDArrayIndex idx : indices){
            if(idx instanceof SpecifiedIndex){
                isSpecifiedIndex = true;
                break;
            }
        }

        if(!isSpecifiedIndex){
            return target.put(indices, source);
        } else {
            //Can't get a view, so we'll do it in subsets instead
            // This is inefficient, but it is correct...
            int numSpecified = 0;
            List<long[]> specifiedIdxs = new ArrayList<>();
            List<Integer> specifiedIdxDims = new ArrayList<>();

            INDArrayIndex[] destinationIndices = indices.clone();  //Shallow clone
            INDArrayIndex[] sourceIndices = indices.clone();
            for( int i=0; i<indices.length; i++){
                INDArrayIndex idx = indices[i];
                if(idx instanceof SpecifiedIndex){
                    numSpecified++;
                    long[] idxs = ((SpecifiedIndex) idx).getIndexes();
                    specifiedIdxs.add(idxs);
                    specifiedIdxDims.add(i);
                } else if(idx instanceof PointIndex){
                    //Example: [2,3,3].put(point(1), ..., [1,x,y]) -> can't use point(1) on [1,x,y]
                    sourceIndices[i] = NDArrayIndex.point(0);
                }
            }
            int[] counts = new int[specifiedIdxs.size()];
            int[] dims = new int[specifiedIdxDims.size()];
            for( int i=0; i<specifiedIdxs.size(); i++ ){
                counts[i] = specifiedIdxs.get(i).length;
                dims[i] = specifiedIdxDims.get(i);
            }

            NdIndexIterator iter = new NdIndexIterator(counts);
            while(iter.hasNext()){
                long[] iterationIdxs = iter.next();
                for(int i=0; i<iterationIdxs.length; i++ ){
                    long[] indicesForDim = specifiedIdxs.get(i);
                    destinationIndices[dims[i]] = NDArrayIndex.point(indicesForDim[(int)iterationIdxs[i]]);
                    sourceIndices[dims[i]] = NDArrayIndex.point(iterationIdxs[i]);
                }

                INDArray sourceView = source.get(sourceIndices);
                INDArray destinationView = target.get(destinationIndices);
                destinationView.assign(sourceView);
            }
        }
        return target;
    }


}
