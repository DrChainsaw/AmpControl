package ampcontrol.model.training.model.mutate;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.stream.IntStream;

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
        final int[] shapeSource = {1, 6, 2, 2};
        final int[] shapeTarget = {1, 3, 2, 2};
        final int nrofElemsSource = IntStream.of(shapeSource).reduce((i1, i2) -> i1 * i2).getAsInt();
        final INDArray source = Nd4j.linspace(0, nrofElemsSource - 1, nrofElemsSource).reshape(shapeSource);
        final INDArray target = Nd4j.create(shapeTarget);
        final Prune.ReshapeSet reshapeSet = new Prune.ReshapeSet(target, source);
        final Prune prune = new Prune();
        prune.apply(reshapeSet);

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
        System.out.println("shape: " + Arrays.toString(shape));
        shape[0] -= 2;
        shape[2] -= 1;
        final INDArray target = Nd4j.create(shape);
        final Prune.ReshapeSet reshapeSet = new Prune.ReshapeSet(target, source);
        final Prune prune = new Prune();
        prune.apply(reshapeSet);

        final INDArray expected = Nd4j.create(new double[][][][]{
                {{{6}}, {{4}}, {{2}}},
                {{{3}}, {{2}}, {{0}}},
        });

        assertEquals("Incorrect output!", expected, target);
    }

    @Test
    public void testShapeCorruption() {
        final INDArray arr = Nd4j.create(new double[][]{{1, 2}, {3, 4}});
        final long shape0 = arr.shape()[0];
        arr.shape()[0] -= 1;
        assertEquals("Incorrect shape!", shape0, arr.shape()[0]);
    }
}