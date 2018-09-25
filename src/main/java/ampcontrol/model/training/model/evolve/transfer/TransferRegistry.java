package ampcontrol.model.training.model.evolve.transfer;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

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
    private int runningNr = 0;

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
                // Not just (premature) optimization, this also seems to avoid some exceptions, possibly due to dl4j issue #6327
                if (indexMap.isEmpty()) {
                    array.assign(anotherArray);
                    return;
                }

                array.put(asIndArray(), anotherArray);
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
        return register(array, String.valueOf(runningNr++));
    }

    ArrayEntry register(INDArray array, String debugName) {
        return registry.computeIfAbsent(Objects.requireNonNull(array, "Got null array for " + debugName), arr -> new ArrayEntry(array, debugName));
    }

    void commit() {
        actions.values().forEach(Runnable::run);
    }
}
