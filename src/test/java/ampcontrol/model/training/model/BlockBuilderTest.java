package ampcontrol.model.training.model;

import ampcontrol.model.training.model.builder.BlockBuilder;
import ampcontrol.model.training.model.layerblocks.*;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

/**
 * Test cases for {@link BlockBuilder}. Class is probably too generic to test and it is very difficult to assess if a
 * created model is "correct" apart from not crashing.
 *
 * @author Christian Skärby
 */
public class BlockBuilderTest {

    /**
     * Test that a very simple model can be built without crashing
     */
    @Test
    public void build() {
        final int nrofLayers = new BlockBuilder()
                .first(new ConvType(new int[]{4, 4, 1})) // Must be at least equal to default kernel size
                .andThen(new Conv2D())
                .andFinally(new Output(1))
                .build().getnLayers();
        assertEquals("Incorrect number of layers!", 2, nrofLayers);
    }

    /**
     * Test that a very simple model can be built without crashing
     */
    @Test
    public void buildRes() {
        final int nrofLayers = new BlockBuilder()
                .first(new ConvType(new int[]{3, 3, 64}))
                .andThenRes()
                .of(new Act()) // Just something which preserves input size
                .andFinally(new Output(1))
                .buildGraph().getVertices().length;
        assertEquals("Incorrect number of layers!", 4, nrofLayers);
    }

    /**
     * Test that a very simple model can be built without crashing
     */
    @Test
    public void buildResAgg() {
        final int nrofLayers = new BlockBuilder()
                .first(new ConvType(new int[]{3, 3, 64})) // Must be at least equal to kernel size
                .andThenRes()
                .aggOf(new Conv2D().setKernelSize(3))
                .andFinally(new ZeroPad().setPad(1))
                .andFinally(new Output(1))
                .buildGraph().getVertices().length;
        assertEquals("Incorrect number of layers!", 5, nrofLayers);
    }

    /**
     * Test that a very simple model can be built without crashing
     */
    @Test
    public void buildResStack() {
        final int nrofStacks = 3;
        final int nrofLayers = new BlockBuilder()
                .first(new ConvType(new int[]{3, 3, 64}))
                .andThenRes()
                .ofStack(nrofStacks)
                .of(new Act()) // Just something which preserves input size
                .andFinally(new Output(1))
                .buildGraph().getVertices().length;
        assertEquals("Incorrect number of layers!", nrofStacks + 3, nrofLayers);
    }

    /**
     * Test that a very simple model can be built without crashing
     */
    @Test
    public void buildDense() {
        final int nrofStacks = 3;
        final int nrofLayers = new BlockBuilder()
                .first(new ConvType(new int[]{3, 3, 64}))
                .andThenDenseStack(nrofStacks)
                .of(new Act()) // Just something which preserves input size
                .andFinally(new Output(1))
                .buildGraph().getVertices().length;
        assertEquals("Incorrect number of layers!", nrofStacks * 2 + 2, nrofLayers);
    }

    /**
     * Test that a very simple model can be built without crashing
     */
    @Test
    public void buildDenseAgg() {
        final int nrofStacks = 3;
        final int nrofLayers = new BlockBuilder()
                .first(new ConvType(new int[]{3, 3, 64})) // Must be at least equal to kernel size
                .andThenDenseStack(nrofStacks)
                .aggOf(new Conv2D().setKernelSize(3))
                .andFinally(new ZeroPad().setPad(1))
                .andFinally(new Output(1))
                .buildGraph().getVertices().length;
        assertEquals("Incorrect number of layers!", nrofStacks * 3 + 2, nrofLayers);
    }

    /**
     * Test that a very simple model can be built without crashing
     */
    @Test
    public void buildDenseAggStack() {
        final int nrofStacks = 3;
        final int nrofStacksInEachStack = 5;
        final int nrofLayers = new BlockBuilder()
                .first(new ConvType(new int[]{3, 3, 64}))
                .andThenDenseStack(nrofStacks)
                .aggStack(nrofStacksInEachStack)
                .of(new Act()) // Just something which preserves input size
                .andFinally(new Act()) // Just something which preserves input size
                .andFinally(new Output(1))
                .buildGraph().getVertices().length;
        assertEquals("Incorrect number of layers!", nrofStacks * 2 + nrofStacks * nrofStacksInEachStack + 2, nrofLayers);
    }

    /**
     * Test that a very simple model can be built without crashing
     */
    @Test
    public void buildStack() {
        final int nrofStacks = 3;
        final int nrofLayers = new BlockBuilder()
                .first(new ConvType(new int[]{nrofStacks * 4, nrofStacks * 4, 1})) // Must be at least equal to default kernel size
                .andThenStack(nrofStacks)
                .of(new Conv2D())
                .andFinally(new Output(1))
                .build().getnLayers();
        assertEquals("Incorrect number of layers!", nrofStacks + 1, nrofLayers);
    }

    /**
     * Test that a very simple model can be built without crashing
     */
    @Test
    public void buildStackAgg() {
        final int nrofStacks = 3;
        final int nrofLayers = new BlockBuilder()
                .first(new ConvType(new int[]{nrofStacks * 4, nrofStacks * 4, 1})) // Must be at least equal to default kernel size
                .andThenStack(nrofStacks)
                .aggOf(new Conv2D())
                .andFinally(new ZeroPad())
                .andFinally(new Output(1))
                .build().getnLayers();
        assertEquals("Incorrect number of layers!", 2 * nrofStacks + 1, nrofLayers);
    }

    /**
     * Test that a very simple model can be built without crashing
     */
    @Test
    public void buildStackStack() {
        final int nrofStacks = 3;
        final int nrofStacksInEachStack = 5;
        final int nrofLayers = new BlockBuilder()
                .first(new ConvType(new int[]{4, 4, 1}))
                .andThenStack(nrofStacks)
                .aggStack(nrofStacksInEachStack)
                .of(new Act()) // Just something which preserves input size
                .andFinally(new Act()) // Just something which preserves input size
                .andFinally(new Output(1))
                .build().getnLayers();
        assertEquals("Incorrect number of layers!", nrofStacks + nrofStacks * nrofStacksInEachStack + 1, nrofLayers);
    }

    /**
     * Test that a very simple model can be built without crashing
     */
    @Test
    public void buildStackRes() {
        final int nrofStacks = 3;
        final int nrofLayers = new BlockBuilder()
                .first(new ConvType(new int[]{4, 4, 1}))
                .andThenStack(nrofStacks)
                .res()
                .of(new Act()) // Just something which preserves input size
                .andFinally(new Output(1))
                .buildGraph().getVertices().length;
        assertEquals("Incorrect number of layers!", 2 * nrofStacks + 2, nrofLayers);
    }

    /**
     * Test that a very simple model can be built without crashing
     */
    @Test
    public void buildStackAggRes() {
        final int nrofStacks = 3;
        final int nrofLayers = new BlockBuilder()
                .first(new ConvType(new int[]{4, 4, 1}))
                .andThenStack(nrofStacks)
                .aggRes()
                .of(new Act()) // Just something which preserves input size
                .andFinally(new Act()) // Just something which preserves input size
                .andFinally(new Output(1))
                .buildGraph().getVertices().length;
        assertEquals("Incorrect number of layers!", 3 * nrofStacks + 2, nrofLayers);
    }

    /**
     * Test that a very simple model can be built without crashing
     */
    @Test
    public void buildStackAggDense() {
        final int nrofStacks = 3;
        final int nrofStacksInEachStack = 5;
        final int nrofLayers = new BlockBuilder()
                .first(new ConvType(new int[]{nrofStacks * 4, nrofStacks * 4, 1})) // Must be at least equal to default kernel size
                .andThenStack(nrofStacks)
                .aggDenseStack(nrofStacksInEachStack)
                .of(new Act()) // Just something which preserves input size
                .andFinally(new Act()) // Just something which preserves input size
                .andFinally(new Output(1))
                .buildGraph().getVertices().length;
        assertEquals("Incorrect number of layers!",  nrofStacks * (2*nrofStacksInEachStack + 1) + 2, nrofLayers);
    }

    /**
     * Test that a very simple model can be built without crashing
     */
    @Test
    public void buildMulti() {
        final int nrofLayers = new BlockBuilder()
                .first(new ConvType(new int[]{3, 3, 64}))
                .multiLevel()
                .andThen(new Act()) // Just something which preserves input size
                .done()
                .andFinally(new Output(1))
                .buildGraph().getVertices().length;
        assertEquals("Incorrect number of layers!", 7, nrofLayers);
    }

    /**
     * Test that a very simple model can be built without crashing
     */
    @Test
    public void buildMultiAgg() {
        final int nrofLayers = new BlockBuilder()
                .first(new ConvType(new int[]{3, 3, 64}))
                .multiLevel()
                .andThenAgg(new Act()) // Just something which preserves input size
                .andFinally(new Act()) // Just something which preserves input size
                .done()
                .andFinally(new Output(1))
                .buildGraph().getVertices().length;
        assertEquals("Incorrect number of layers!", 8, nrofLayers);
    }

    /**
     * Test that a very simple model can be built without crashing
     */
    @Test
    public void buildMultiRes() {
        final int nrofLayers = new BlockBuilder()
                .first(new ConvType(new int[]{3, 3, 64}))
                .multiLevel()
                .andThenRes()
                .of(new Act()) // Just something which preserves input size
                .done()
                .andFinally(new Output(1))
                .buildGraph().getVertices().length;
        assertEquals("Incorrect number of layers!", 8, nrofLayers);
    }

    /**
     * Test that a very simple model can be built without crashing
     */
    @Test
    public void buildMultiAggRes() {
        final int nrofLayers = new BlockBuilder()
                .first(new ConvType(new int[]{3, 3, 64}))
                .multiLevel()
                .andThenAggRes()
                .of(new Act()) // Just something which preserves input size
                .andFinally(new Act())
                .done()
                .andFinally(new Output(1))
                .buildGraph().getVertices().length;
        assertEquals("Incorrect number of layers!", 9, nrofLayers);
    }

    /**
     * Test that a very simple model can be built without crashing
     */
    @Test
    public void buildMultiStack() {
        final int nrofStacks = 3;
        final int nrofLayers = new BlockBuilder()
                .first(new ConvType(new int[]{3, 3, 64}))
                .multiLevel()
                .andThenStack(nrofStacks)
                .of(new Act()) // Just something which preserves input size
                .done()
                .andFinally(new Output(1))
                .buildGraph().getVertices().length;
        assertEquals("Incorrect number of layers!", 6 + nrofStacks, nrofLayers);
    }

    /**
     * Test that a very simple model can be built without crashing
     */
    @Test
    public void buildResFork() {
        final int nrofChannels = 64;
        final int nrofLayers = new BlockBuilder()
                .first(new ConvType(new int[]{3, 3, nrofChannels}))
                .andThenRes()
                .ofFork()
                .add(new Conv2D().setKernelSize(1).setNrofKernels(nrofChannels / 2))
                .add(new Conv2D().setKernelSize(1).setNrofKernels(nrofChannels / 2))
                .done()
                .andFinally(new Output(1))
                .buildGraph().getVertices().length;
        assertEquals("Incorrect number of layers!", 6, nrofLayers);
    }

    /**
     * Test that a very simple model can be built without crashing
     */
    @Test
    public void buildResAggFork() {
        final int nrofChannels = 64;
        final int nrofLayers = new BlockBuilder()
                .first(new ConvType(new int[]{3, 3, nrofChannels}))
                .andThenRes()
                .aggFork()
                .add(new Conv2D().setKernelSize(1).setNrofKernels(nrofChannels*2))
                .add(new Conv2D().setKernelSize(1).setNrofKernels(nrofChannels*2))
                .done()
                .andFinally(new Conv2D().setKernelSize(1).setNrofKernels(nrofChannels))
                .andFinally(new Output(1))
                .buildGraph().getVertices().length;
        assertEquals("Incorrect number of layers!", 7, nrofLayers);
    }

    /**
     * Test that a very simple model can be built without crashing
     */
    @Test
    public void buildResForkAgg() {
        final int nrofChannels = 64;
        final int nrofLayers = new BlockBuilder()
                .first(new ConvType(new int[]{3, 3, nrofChannels}))
                .andThenRes()
                .ofFork()
                .addAgg(new Conv2D().setKernelSize(1).setNrofKernels(nrofChannels * 2))
                .andFinally(new Conv2D().setKernelSize(1).setNrofKernels(nrofChannels / 2))
                .add(new Conv2D().setKernelSize(1).setNrofKernels(nrofChannels / 2))
                .done()
                .andFinally(new Output(1))
                .buildGraph().getVertices().length;
        assertEquals("Incorrect number of layers!", 7, nrofLayers);
    }

    /**
     * Test that a very simple model can be built without crashing
     */
    @Test
    public void buildResForkStack() {
        final int nrofChannels = 64;
        final int nrofLayers = new BlockBuilder()
                .first(new ConvType(new int[]{3, 3, nrofChannels}))
                .andThenRes()
                .ofFork()
                .addStack(2)
                .of(new Conv2D().setKernelSize(1).setNrofKernels(nrofChannels / 2))
                .add(new Conv2D().setKernelSize(1).setNrofKernels(nrofChannels / 2))
                .done()
                .andFinally(new Output(1))
                .buildGraph().getVertices().length;
        assertEquals("Incorrect number of layers!", 7, nrofLayers);
    }

}