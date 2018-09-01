package ampcontrol.model.training.model.mutate.reshape;

import com.google.common.primitives.Ints;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.Comparator;

import static junit.framework.TestCase.assertEquals;

/**
 * Test cases for {@link Prune}
 *
 * @author Christian Skärby
 */
public class PruneTest {

    /**
     * Test pruning in one dimension out of four
     */
    @Test
    public void applyOneDimension() {
        final long[] shapeSource = {1, 6, 2, 2};
        final long[] shapeTarget = {1, 3, 2, 2};
        final INDArray source = createLinspace(shapeSource);
        final INDArray target = Nd4j.create(shapeTarget);
        final ReshapeInstruction reshapeInstruction = ReshapeInstruction.builder()
                .targetShape(shapeTarget)
                .sourceShape(shapeSource)
                .pruneInstruction(Prune.SinglePruneInstruction.builder()
                        .source(source)
                        .target(target)
                        .build())
                .build();
        final Prune prune = new Prune();
        prune.accept(reshapeInstruction);

        reshapeInstruction.reshape();
        final INDArray expected = Nd4j.create(new double[][][][]{{
                {
                        {12, 13},
                        {14, 15}
                },
                {
                        {16, 17},
                        {18, 19}
                },
                {
                        {20, 21},
                        {22, 23}
                }
        }});

        assertEquals("Incorrect output!", expected, target);
    }

    /**
     * Test pruning in two dimension out of four
     */
    @Test
    public void applyTwoDimension() {
        final INDArray source = Nd4j.create(new double[][][][]{
                {{{6}, {5}}, {{4}, {3}}, {{2}, {1}}}, // = 21 => keep elem 0 in dim 0
                {{{0}, {1}}, {{2}, {3}}, {{4}, {5}}}, // = 15
                {{{6}, {0}}, {{5}, {1}}, {{4}, {2}}}, // = 18
                {{{3}, {4}}, {{2}, {5}}, {{0}, {6}}}  // = 20 => keep elem 3 in dim 0
        });  //   15   10     13   12     10   14
        // 15 + 13 + 10 = 38 => keep elem 0 in dim 2
        // 10 + 12 + 14 = 36

        final long[] shape = source.shape().clone();
        shape[0] -= 2;
        shape[2] -= 1;
        final INDArray target = Nd4j.create(shape);
        final ReshapeInstruction reshapeInstruction = ReshapeInstruction.builder()
                .targetShape(target.shape())
                .sourceShape(source.shape())
                .pruneInstruction(Prune.SinglePruneInstruction.builder()
                        .source(source)
                        .target(target)
                        .build())
                .build();
        final Prune prune = new Prune();
        prune.accept(reshapeInstruction);

        reshapeInstruction.reshape();
        final INDArray expected = Nd4j.create(new double[][][][]{
                {{{6}}, {{4}}, {{2}}},
                {{{3}}, {{2}}, {{0}}},
        });

        assertEquals("Incorrect output!", expected, target);
    }

    /**
     * Test pruning of one array which is coupled to another array.
     */
    @Test
    public void applyCoupled() {
        final long[] shapeSource = {2, 6, 3, 3}; // dim 1 is coupled to dim 0 in output
        final long[] shapeTarget = {2, 4, 3, 3}; // dim 1 is coupled to dim 0 in output
        final long[] shapeSourceOutput = {6, 5, 3, 3};
        final long[] shapeTargetOutput = {4, 5, 3, 3};

        // Reduce the number of outputs from 6 to 4, which in turn means that dim 0 in
        // sourceOutput shall be reduced as well.
        // Furthermore, the same element indexes removed from source dim 1 shall be removed from sourceOutput dim 0
        final int[] orderToKeep = {0, 2, 4, 5, 3, 1}; // Note first 4 indexes must be in order for testcase to pass
        final INDArray source = createLinspace(shapeSource);
        final INDArray sourceOutput = createLinspace(shapeSourceOutput);

        final INDArray target = Nd4j.create(shapeTarget);
        final INDArray targetOutput = Nd4j.create(shapeTargetOutput);

        final ReshapeInstruction reshapeInstruction = ReshapeInstruction.builder()
                .sourceShape(shapeSource)
                .targetShape(shapeTarget)
                .pruneInstruction(
                        Prune.PruneInstructionList.builder()
                                .instruction(Prune.SinglePruneInstruction.builder()
                                        .compFactory(dummy -> Comparator.comparingInt(i -> Ints.indexOf(orderToKeep, i)))
                                        .source(source)
                                        .target(target)
                                        .build())
                                .instruction(Prune.SinglePruneInstruction.builder()
                                        .source(sourceOutput)
                                        .target(targetOutput)
                                        .dimensionMapper(dim -> dim == 1 ? 0 : (dim == 0 ? 1 : dim))
                                        .build())

                                .build())
                .build();

        final Prune prune = new Prune();
        prune.accept(reshapeInstruction);

        reshapeInstruction.reshape();
        for (int elemInd = 0; elemInd < shapeTarget[1]; elemInd++) {
            assertEquals("Incorrect target for element index " + elemInd + "!",
                    target.tensorAlongDimension(elemInd, 0, 2, 3),
                    source.tensorAlongDimension(orderToKeep[elemInd], 0, 2, 3));

        }

        for (int elemInd = 0; elemInd < shapeTarget[0]; elemInd++) {
            assertEquals("Incorrect target output for element index " + elemInd + "!",
                    targetOutput.tensorAlongDimension(elemInd, 1, 2, 3),
                    sourceOutput.tensorAlongDimension(orderToKeep[elemInd], 1, 2, 3));

        }
    }

    private static INDArray createLinspace(long[] shapeSource) {
        final long nrofElemsSource = Arrays.stream(shapeSource).reduce((i1, i2) -> i1 * i2).orElseThrow(() -> new IllegalArgumentException("No elements!"));
        return Nd4j.linspace(0, nrofElemsSource - 1, nrofElemsSource).reshape(shapeSource);
    }
}