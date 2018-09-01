package ampcontrol.model.training.model.mutate.reshape;

import com.google.common.primitives.Ints;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.Comparator;
import java.util.function.IntUnaryOperator;
import java.util.stream.IntStream;

import static junit.framework.TestCase.assertEquals;

/**
 * Test cases for {@link ReshapeTask}
 *
 * @author Christian Skärby
 */
public class ReshapeTaskTest {

    /**
     * Test pruning in one dimension out of four
     */
    @Test
    public void applySizeDecrease() {
        final long[] shapeSource = {1, 6, 2, 2};
        final long[] shapeTarget = {1, 3, 2, 2};
        final INDArray source = createLinspace(shapeSource);
        final INDArray target = Nd4j.create(shapeTarget);
        final ReshapeTask reshapeTask = ReshapeTask.builder()
                .targetShape(shapeTarget)
                .sourceShape(shapeSource)
                .pruneInstruction(SingleReshapeSubTask.builder()
                        .source(source)
                        .target(target)
                        .build())
                .build();

        reshapeTask.reshape();
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
    public void applySizeDecreaseTwoDimension() {
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
        final ReshapeTask reshapeTask = ReshapeTask.builder()
                .targetShape(target.shape())
                .sourceShape(source.shape())
                .pruneInstruction(SingleReshapeSubTask.builder()
                        .source(source)
                        .target(target)
                        .build())
                .build();

        reshapeTask.reshape();
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
    public void applySizeDecreaseCoupled() {
        final long[] shapeSource = {2, 6, 3, 3}; // dim 1 is coupled to dim 0 in output
        final long[] shapeTarget = {2, 4, 3, 3}; // dim 1 is coupled to dim 0 in output
        final long[] shapeSourceOutput = {6, 5, 3, 3};
        final long[] shapeTargetOutput = {4, 5, 3, 3};

        // Reduce the number of outputs from 6 to 4, which in turn means that dim 0 in
        // sourceOutput shall be reduced as well.
        // Furthermore, the same element indexes removed from source dim 1 shall be removed from sourceOutput dim 0
        final int[] orderToKeep = {0, 2, 4, 5, 3, 1}; // Note: first 4 indexes must be in order for testcase to pass
        final INDArray source = createLinspace(shapeSource);
        final INDArray sourceOutput = createLinspace(shapeSourceOutput);

        final INDArray target = Nd4j.create(shapeTarget);
        final INDArray targetOutput = Nd4j.create(shapeTargetOutput);

        final ReshapeTask reshapeTask = ReshapeTask.builder()
                .sourceShape(shapeSource)
                .targetShape(shapeTarget)
                .pruneInstruction(
                        ReshapeSubTaskList.builder()
                                .instruction(SingleReshapeSubTask.builder()
                                        .compFactory(dummy -> Comparator.comparingInt(i -> Ints.indexOf(orderToKeep, i)))
                                        .source(source)
                                        .target(target)
                                        .build())
                                .instruction(SingleReshapeSubTask.builder()
                                        .source(sourceOutput)
                                        .target(targetOutput)
                                        .sourceIndMapping(SingleReshapeSubTask.IndMapping.builder()
                                                .dimensionMapper(dim -> dim == 1 ? 0 : (dim == 0 ? 1 : dim)) // Swap dim 0 and 1
                                                .build())
                                        .build())

                                .build())
                .build();


        reshapeTask.reshape();
        for (int elemInd = 0; elemInd < shapeTarget[1]; elemInd++) {
            assertEquals("Incorrect target for element index " + elemInd + "!",
                    source.tensorAlongDimension(orderToKeep[elemInd], 0, 2, 3),
                    target.tensorAlongDimension(elemInd, 0, 2, 3));

        }

        for (int elemInd = 0; elemInd < shapeTarget[0]; elemInd++) {
            assertEquals("Incorrect target output for element index " + elemInd + "!",
                    sourceOutput.tensorAlongDimension(orderToKeep[elemInd], 1, 2, 3),
                    targetOutput.tensorAlongDimension(elemInd, 1, 2, 3));

        }
    }

    /**
     * Test increasing the size of one array which is coupled to another array.
     */
    @Test
    public void applySizeIncrease() {
        final long[] shapeSource = {2, 4, 3, 3};
        final long[] shapeTarget = {2, 6, 3, 3};


        final INDArray source = createLinspace(shapeSource);
        final INDArray target = Nd4j.create(shapeTarget);

        final ReshapeTask reshapeTask = ReshapeTask.builder()
                .sourceShape(shapeSource)
                .targetShape(shapeTarget)
                .pruneInstruction(SingleReshapeSubTask.builder()
                        .source(source)
                        .target(target)
                        .build())
                .build();

        reshapeTask.reshape();
        for (int elemInd = 0; elemInd < shapeSource[1]; elemInd++) {
            assertEquals("Incorrect target for element index " + elemInd + "!",
                    source.tensorAlongDimension(elemInd, 0, 2, 3),
                    target.tensorAlongDimension(elemInd, 0, 2, 3));
        }
    }

    /**
     * Test increasing the size
     */
    @Test
    public void applySizeIncreaseCoupled() {
        final long[] shapeSource = {2, 4, 3, 3}; // dim 1 is coupled to dim 0 in output
        final long[] shapeTarget = {2, 6, 3, 3}; // dim 1 is coupled to dim 0 in output
        final long[] shapeSourceOutput = {4, 5, 3, 3};
        final long[] shapeTargetOutput = {6, 5, 3, 3};

        // Reduce the number of outputs from 6 to 4, which in turn means that dim 0 in
        // sourceOutput shall be reduced as well.
        // Furthermore, the same element indexes removed from source dim 1 shall be removed from sourceOutput dim 0
        final INDArray source = createLinspace(shapeSource);
        final INDArray sourceOutput = createLinspace(shapeSourceOutput);

        final INDArray target = Nd4j.ones(shapeTarget);
        final INDArray targetOutput = Nd4j.ones(shapeTargetOutput);

        final ReshapeTask reshapeTask = ReshapeTask.builder()
                .sourceShape(shapeSource)
                .targetShape(shapeTarget)
                .pruneInstruction(
                        ReshapeSubTaskList.builder()
                                .instruction(SingleReshapeSubTask.builder()
                                        .source(source)
                                        .target(target)
                                        .build())
                                .instruction(SingleReshapeSubTask.builder()
                                        .source(sourceOutput)
                                        .target(targetOutput)
                                        .build())
                                .build())
                .build();

        reshapeTask.reshape();

        for (int elemInd = 0; elemInd < shapeSource[1]; elemInd++) {
            assertEquals("Incorrect target for element index " + elemInd + "!",
                    source.tensorAlongDimension(elemInd, 0, 2, 3),
                    target.tensorAlongDimension(elemInd, 0, 2, 3));

        }

        for (int elemInd = 0; elemInd < shapeSource[0]; elemInd++) {
            assertEquals("Incorrect target output for element index " + elemInd + "!",
                    sourceOutput.tensorAlongDimension(elemInd, 1, 2, 3),
                    targetOutput.tensorAlongDimension(elemInd, 1, 2, 3));

        }
    }

    /**
     * Test simultaneous increasing and decrease of the size
     */
    @Test
    public void applySizeIncreaseAndDecrease() {
        final long[] shapeSource = {4, 4, 3, 3};
        final long[] shapeTarget = {2, 6, 3, 3};


        final INDArray source = createLinspace(shapeSource);
        final INDArray target = Nd4j.ones(shapeTarget);

        final int[] orderToKeep = {0, 2, 1, 3}; // Note: first 2 indexes must be in order for testcase to pass
        final int dimOneElemOffset = 1;
        final ReshapeTask reshapeTask = ReshapeTask.builder()
                .sourceShape(shapeSource)
                .targetShape(shapeTarget)
                .pruneInstruction(SingleReshapeSubTask.builder()
                        .compFactory(dummy -> Comparator.comparingInt(i -> Ints.indexOf(orderToKeep, i)))
                        .source(source)
                        .target(target)
                        .targetIndMapping(SingleReshapeSubTask.IndMapping.builder()
                                .remapper(dim -> dim == 1 ? elem -> elem + dimOneElemOffset : IntUnaryOperator.identity())
                                .build())
                        .build())
                .build();

        reshapeTask.reshape();
        for (int elemInd0 = 0; elemInd0 < shapeTarget[0]; elemInd0++) {
            for (int elemInd1 = 0; elemInd1 < shapeSource[1]; elemInd1++) {
                assertEquals("Incorrect target for element index " + elemInd0 + " + " + elemInd1 + "!",
                        source.tensorAlongDimension(elemInd1, 0, 2, 3).tensorAlongDimension(orderToKeep[elemInd0], 1, 2),
                        target.tensorAlongDimension(elemInd1 + dimOneElemOffset, 0, 2, 3).tensorAlongDimension(elemInd0, 1, 2));
            }
        }

        IntStream.concat(IntStream.range(0, dimOneElemOffset), IntStream.range((int) shapeSource[1] + dimOneElemOffset, (int) shapeTarget[1]))
                .forEach(elemInd1 ->
                        assertEquals("Expected array unchanged!",
                                1d,
                                target.tensorAlongDimension(elemInd1, 0, 2, 3).meanNumber().doubleValue(), 1e-10));

    }

    private static INDArray createLinspace(long[] shapeSource) {
        final long nrofElemsSource = Arrays.stream(shapeSource).reduce((i1, i2) -> i1 * i2).orElseThrow(() -> new IllegalArgumentException("No elements!"));
        return Nd4j.linspace(0, nrofElemsSource - 1, nrofElemsSource).reshape(shapeSource);
    }
}