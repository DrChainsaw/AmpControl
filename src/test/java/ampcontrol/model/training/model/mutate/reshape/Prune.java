package ampcontrol.model.training.model.mutate.reshape;

import lombok.Builder;
import lombok.Singular;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.indexing.SpecifiedIndex;

import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.function.IntUnaryOperator;
import java.util.stream.IntStream;

/**
 * Prunes a target {@link INDArray} if it has fewer dimensions than a source {@link INDArray} and stores the result
 * in the target.
 *
 * @author Christian Skärby
 */
public class Prune implements Consumer<ReshapeInstruction> {

    /**
     * An instruction for how to prune weights of a layer, including inputs to subsequent layers
     */
    public interface PruneInstruction {
        /**
         * Adds element indexes in a given dimension which shall be kept from a source.
         *
         * @param dim               wanted dimension
         * @param wantedElementInds wanted element indexes in given dimension
         */
        void addWantedElements(int dim, int[] wantedElementInds);

        /**
         * Decides how to compare elements of the given dimensions
         *
         * @param tensorDimensions dimensions to compare
         * @return a Comparator which can compare elements of the given dimensions
         */
        Comparator<Integer> getComparator(int[] tensorDimensions);

        /**
         * Assigns the pruned weights
         */
        void assign();
    }

    @Builder
    public static class PruneInstructionList implements PruneInstruction {
        @Singular
        private final List<PruneInstruction> instructions;

        @Override
        public void addWantedElements(int dim, int[] wantedElementInds) {
            instructions.forEach(instr -> instr.addWantedElements(dim, wantedElementInds));
        }

        @Override
        public Comparator<Integer> getComparator(int[] tensorDimensions) {
            return instructions.get(0).getComparator(tensorDimensions);
        }

        @Override
        public void assign() {
            instructions.forEach(PruneInstruction::assign);
        }
    }

    @Builder(builderClassName = "Builder")
    public static class SinglePruneInstruction implements PruneInstruction {
        private final INDArray source;
        private final INDArray target;
        private INDArrayIndex[] wantedIndsPerDim;
        private final Function<int[], Comparator<Integer>> compFactory;
        @lombok.Builder.Default
        private final IntUnaryOperator dimensionMapper = IntUnaryOperator.identity();
        @lombok.Builder.Default
        private final Function<int[], int[]> remapper = Function.identity();

        @Override
        public void addWantedElements(int dim, int[] wantedElementInds) {
            if (wantedIndsPerDim == null) {
                wantedIndsPerDim = new INDArrayIndex[source.rank()];
                Arrays.fill(wantedIndsPerDim, NDArrayIndex.all());
            }
            wantedIndsPerDim[dimensionMapper.applyAsInt(dim)] =
                    new SpecifiedIndex(remapper.apply(wantedElementInds));
        }

        @Override
        public Comparator<Integer> getComparator(int[] tensorDimensions) {
            return compFactory.apply(tensorDimensions);
        }

        @Override
        public void assign() {
            target.assign(source.get(wantedIndsPerDim));
        }

        public static class Builder {
            // Used through lombok
            private Function<int[], Comparator<Integer>> compFactory = tensorDimensions -> (e1, e2) -> -Double.compare(
                    source.tensorAlongDimension(e1, tensorDimensions).sumNumber().doubleValue(),
                    source.tensorAlongDimension(e2, tensorDimensions).sumNumber().doubleValue());
        }

    }

    @Override
    public void accept(ReshapeInstruction reshapeInstruction) {
        final long[] shapeTarget = reshapeInstruction.getTargetShape();
        final long[] shapeSource = reshapeInstruction.getSourceShape();
        if (shapeTarget.length != shapeSource.length) {
            throw new IllegalArgumentException("Must have same dimensions! shapeTarget: " + Arrays.toString(shapeTarget) + " shapeSource: " + Arrays.toString(shapeSource));
        }

        final PruneInstruction pruneInstruction = reshapeInstruction.getPruneInstruction();
        for (int i = 0; i < shapeTarget.length; i++) {
            if (shapeTarget[i] < shapeSource[i]) {
                final int tensorDim = i;
                final int[] tensorDimensions = IntStream.range(0, shapeTarget.length)
                        .filter(dim -> tensorDim != dim)
                        .toArray();

                final Comparator<Integer> comparator = pruneInstruction.getComparator(tensorDimensions);
                final int[] wantedElements = IntStream.range(0, (int) shapeSource[tensorDim])
                        .boxed()
                        .sorted(comparator)
                        .limit(shapeTarget[tensorDim])
                        .mapToInt(e -> e)
                        .sorted()
                        .toArray();

                pruneInstruction.addWantedElements(tensorDim, wantedElements);
            }
        }
    }
}
