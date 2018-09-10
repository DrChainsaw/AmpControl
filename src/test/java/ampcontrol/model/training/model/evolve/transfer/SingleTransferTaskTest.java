package ampcontrol.model.training.model.evolve.transfer;

import com.google.common.primitives.Ints;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.Comparator;
import java.util.function.Function;
import java.util.function.IntUnaryOperator;
import java.util.stream.IntStream;

import static junit.framework.TestCase.assertEquals;

/**
 * Test cases for {@link SingleTransferTask}
 *
 * @author Christian Skärby
 */
public class SingleTransferTaskTest {

    /**
     * Test pruning in one dimension out of four
     */
    @Test
    public void applySizeDecrease() {
        final long[] shapeSource = {1, 6, 2, 2};
        final long[] shapeTarget = {1, 3, 2, 2};
        final INDArray source = createLinspace(shapeSource);
        final INDArray target = Nd4j.create(shapeTarget);
        final TransferRegistry registry = new TransferRegistry();
        SingleTransferTask.builder()
                .target(SingleTransferTask.IndMapping.builder()
                        .entry(registry.register(target))
                        .build())
                .source(SingleTransferTask.IndMapping.builder()
                        .entry(registry.register(source))
                        .build())
                .build().execute();

        registry.commit();
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
        final TransferRegistry registry = new TransferRegistry();
        SingleTransferTask.builder()
                .target(SingleTransferTask.IndMapping.builder()
                        .entry(registry.register(target))
                        .build())
                .source(SingleTransferTask.IndMapping.builder()
                        .entry(registry.register(source))
                        .build())
                .build().execute();

        registry.commit();
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

        final TransferRegistry registry = new TransferRegistry();
        SingleTransferTask.builder()
                .compFactory(fixedOrderComp(orderToKeep))
                .target(SingleTransferTask.IndMapping.builder()
                        .entry(registry.register(target, "target"))
                        .build())
                .source(SingleTransferTask.IndMapping.builder()
                        .entry(registry.register(source, "source"))
                        .build())
                .addDependentTask(SingleTransferTask.builder()
                        .target(SingleTransferTask.IndMapping.builder()
                                .entry(registry.register(targetOutput, "targetOutput"))
                                .build())
                        .source(SingleTransferTask.IndMapping.builder()
                                .entry(registry.register(sourceOutput, "sourceOutput"))
                                .dimensionMapper(dim -> dim == 1 ? 0 : (dim == 0 ? 1 : dim)) // Swap dim 0 and 1
                                .build()))
                .build()
                .execute();

        registry.commit();
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

        final TransferRegistry registry = new TransferRegistry();

        final INDArray source = createLinspace(shapeSource);
        final INDArray target = Nd4j.create(shapeTarget);

        SingleTransferTask.builder()
                .target(SingleTransferTask.IndMapping.builder()
                        .entry(registry.register(target))
                        .build())
                .source(SingleTransferTask.IndMapping.builder()
                        .entry(registry.register(source))
                        .build())
                .build().execute();

        registry.commit();
        for (int elemInd = 0; elemInd < shapeSource[1]; elemInd++) {
            assertEquals("Incorrect target for element index " + elemInd + "!",
                    source.tensorAlongDimension(elemInd, 0, 2, 3),
                    target.tensorAlongDimension(elemInd, 0, 2, 3));
        }
    }

    /**
     * Test increasing the size of a dependent task
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

        final TransferRegistry registry = new TransferRegistry();
        SingleTransferTask.builder()
                .target(SingleTransferTask.IndMapping.builder()
                        .entry(registry.register(target))
                        .build())
                .source(SingleTransferTask.IndMapping.builder()
                        .entry(registry.register(source))
                        .build())
                .addDependentTask(SingleTransferTask.builder()
                        .target(SingleTransferTask.IndMapping.builder()
                                .entry(registry.register(targetOutput))
                                .dimensionMapper(dim -> dim == 0 ? 1 : dim == 1 ? 0 : dim)
                                .build())
                        .source(SingleTransferTask.IndMapping.builder()
                                .entry(registry.register(sourceOutput))
                                .build()))
                .build().execute();

        registry.commit();
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
        final TransferRegistry registry = new TransferRegistry();
        SingleTransferTask.builder()
                .compFactory(fixedOrderComp(orderToKeep))
                .target(SingleTransferTask.IndMapping.builder()
                        .entry(registry.register(target))
                        .remapper(dim -> dim == 1 ? elem -> elem + dimOneElemOffset : IntUnaryOperator.identity())
                        .build())
                .source(SingleTransferTask.IndMapping.builder()
                        .entry(registry.register(source))
                        .build())
                .build().execute();

        registry.commit();
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

    /**
     * Test size decrease of two different dimensions, once due to input pruning and once
     * due to output pruning.
     */
    @Test
    public void applySizeDecreaseTwiceDifferentInstances() {
        final long[] shapeSource = {5, 1, 2, 2}; // Dim 0 is coupled to dim1 sourceOutput
        final long[] shapeTarget = {3, 1, 2, 2}; // Dim 0 is coupled to dim1 targetOutput, dim0 is decreased
        final long[] shapeSourceOutput = {4, 5, 2, 2}; // Dim 1 is coupled to dim0 in source
        final long[] shapeTargetOutput = {2, 3, 2, 2}; // Dim 1 is coupled to dim0 target, dim0 is decreased compared to sourceOutput
        final INDArray source = createLinspace(shapeSource);
        final INDArray sourceOutput = Nd4j.reverse(createLinspace(shapeSourceOutput));

        final INDArray target = Nd4j.create(shapeTarget);
        final INDArray targetOutput = Nd4j.create(shapeTargetOutput);

        final TransferRegistry registry = new TransferRegistry();
        final int[] orderToKeepFirst = {3, 1, 4, 2, 0};
        final int[] orderToKeepSecond = {0, 2, 3, 1};
        SingleTransferTask.builder()
                .compFactory(fixedOrderComp(orderToKeepFirst))
                .target(SingleTransferTask.IndMapping.builder()
                        .entry(registry.register(target, "target"))
                        .build())
                .source(SingleTransferTask.IndMapping.builder()
                        .entry(registry.register(source, "source"))
                        .build())
                .addDependentTask(SingleTransferTask.builder()
                        .target(SingleTransferTask.IndMapping.builder()
                                .entry(registry.register(targetOutput, "targetOutput"))
                                .build())
                        .source(SingleTransferTask.IndMapping.builder()
                                .entry(registry.register(sourceOutput, "sourceOutput"))
                                .dimensionMapper(dim -> dim == 0 ? 1 : dim == 1 ? 0 : dim)
                                .build()))
                .build().execute();

        SingleTransferTask.builder()
                .maskDim(1)
                .compFactory(fixedOrderComp(orderToKeepSecond))
                .target(SingleTransferTask.IndMapping.builder()
                        .entry(registry.register(targetOutput, "targetShallNotBeThere"))
                        .build())
                .source(SingleTransferTask.IndMapping.builder()
                        .entry(registry.register(sourceOutput, "sourceShallNotBeThere"))
                        .build())
                .build().execute();

        registry.commit();

        final int[] expectedElemsFirst = IntStream.of(orderToKeepFirst).limit(shapeTarget[0]).sorted().toArray();
        for (int elemInd0 = 0; elemInd0 < shapeTarget[0]; elemInd0++) {
            assertEquals("Incorrect target for element index " + elemInd0 + "!",
                    source.tensorAlongDimension(expectedElemsFirst[elemInd0], 1, 2, 3),
                    target.tensorAlongDimension(elemInd0, 1, 2, 3));
        }

        final int[] expectedElemsSecond = IntStream.of(orderToKeepSecond).limit(shapeTargetOutput[0]).sorted().toArray();
        for (int elemInd0 = 0; elemInd0 < shapeTargetOutput[0]; elemInd0++) {
            for (int elemInd1 = 0; elemInd1 < shapeTargetOutput[1]; elemInd1++) {
                // Remember that "expectedElemsFirst" is mapped to dim1 for output
                assertEquals("Incorrect target output for element index " + elemInd0 + ", " + elemInd1 + "!",
                        sourceOutput.tensorAlongDimension(expectedElemsSecond[elemInd0], 1, 2, 3).tensorAlongDimension(expectedElemsFirst[elemInd1], 1, 2),
                        targetOutput.tensorAlongDimension(elemInd0, 1, 2, 3).tensorAlongDimension(elemInd1, 1, 2));
            }
        }

    }

    private static INDArray createLinspace(long[] shapeSource) {
        final long nrofElemsSource = Arrays.stream(shapeSource).reduce((i1, i2) -> i1 * i2).orElseThrow(() -> new IllegalArgumentException("No elements!"));
        return Nd4j.linspace(0, nrofElemsSource - 1, nrofElemsSource).reshape(shapeSource);
    }

    /**
     * Creates a comparator with a predefined index order
     *
     * @param order the desired order
     * @return a Comparator which will sort integers in the given order
     */
    public static Function<int[], Comparator<Integer>> fixedOrderComp(int[] order) {
        return dummy -> Comparator.comparingInt(i -> Ints.indexOf(order, i));
    }
}