package ampcontrol.model.training.model.mutate;

import lombok.Builder;
import lombok.Getter;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.indexing.SpecifiedIndex;

import java.util.Arrays;
import java.util.Comparator;
import java.util.function.Function;
import java.util.stream.IntStream;

/**
 * Prunes a target {@link INDArray} if it has fewer dimensions than a source {@link INDArray} and stores the result
 * in the target.
 *
 * @author Christian Skärby
 */
public class Prune implements Function<Prune.ReshapeSet, Prune.ReshapeSet> {

    @Builder
    @Getter
    public static class ReshapeSet {
        private final INDArray target;
        private final INDArray source;
private final Function<int[], Comparator<Integer>> compFactory = tensorDimensions -> (e1, e2) -> -Double.compare(
        getSource().tensorAlongDimension(e1, tensorDimensions).sumNumber().doubleValue(),
        getSource().tensorAlongDimension(e2, tensorDimensions).sumNumber().doubleValue());
    }

    @Override
    public ReshapeSet apply(ReshapeSet reshapeSet) {
        final long[] shapeTarget = reshapeSet.getTarget().shape();
        final long[] shapeSource = reshapeSet.getSource().shape();
        if (shapeTarget.length != shapeSource.length) {
            throw new IllegalArgumentException("Must have same dimensions! shapeTarget: " + Arrays.toString(shapeTarget) + " shapeSource: " + Arrays.toString(shapeSource));
        }

        final INDArrayIndex[] indVec = new INDArrayIndex[shapeTarget.length];
        Arrays.fill(indVec, NDArrayIndex.all());
        for (int i = 0; i < shapeTarget.length; i++) {
            if (shapeTarget[i] < shapeSource[i]) {
                final int tensorDim = i;
                final int[] tensorDimensions = IntStream.range(0, shapeTarget.length)
                        .filter(dim -> tensorDim != dim)
                        .toArray();

                final Comparator<Integer> comparator = reshapeSet.compFactory.apply(tensorDimensions);
                final int[] wantedElements = IntStream.range(0, (int)shapeSource[tensorDim])
                        .boxed()
                        .sorted(comparator)
                        .limit(shapeTarget[tensorDim])
                        .mapToInt(e -> e)
                        .sorted()
                        .toArray();

                indVec[tensorDim] = new SpecifiedIndex(wantedElements);
//                final INDArrayIndex[] indVec = new INDArrayIndex[shapeTarget.length];
//                Arrays.fill(indVec, NDArrayIndex.all());
//                IntStream.range(0, wantedElements.length).forEach(e -> {
//                    indVec[tensorDim] = NDArrayIndex.point(e);
//                    reshapeSet.getTarget().put(indVec, reshapeSet.getSource().tensorAlongDimension(wantedElements[e], tensorDimensions));
//                });
            }
        }
        reshapeSet.getTarget().assign(reshapeSet.getSource().get(indVec));

        return reshapeSet;
    }
}
