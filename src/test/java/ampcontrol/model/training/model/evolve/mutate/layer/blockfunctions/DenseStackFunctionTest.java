package ampcontrol.model.training.model.evolve.mutate.layer.blockfunctions;

import ampcontrol.model.training.model.evolve.mutate.util.GraphBuilderUtil;
import ampcontrol.model.training.model.layerblocks.Conv2D;
import ampcontrol.model.training.model.layerblocks.LayerBlockConfig;
import ampcontrol.model.training.model.layerblocks.adapters.GraphAdapter;
import ampcontrol.model.training.model.layerblocks.adapters.LayerSpyAdapter;
import org.apache.commons.lang.mutable.MutableInt;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.Convolution2D;
import org.deeplearning4j.nn.conf.layers.FeedForwardLayer;
import org.junit.Test;

import java.util.Arrays;
import java.util.List;
import java.util.function.Function;

import static junit.framework.TestCase.assertEquals;

/**
 * Test cases for {@link DenseStackFunction}
 *
 * @author Christian Skärby
 */
public class DenseStackFunctionTest {

    /**
     * Test that the dense stack has the right number of blocks and the right nOut
     */
    @Test
    public void apply() {
        final List<Long> expectedStackChoices = Arrays.asList( 2L, 3L, 7L);
        final Function<Long, LayerBlockConfig>  denseFunction = new DenseStackFunction(
                stackChoices -> {
                    assertEquals("Incorrect stack choices", expectedStackChoices, stackChoices);
                    return stackChoices.get(1);
                }, nOut -> new Conv2D().setNrofKernels(nOut.intValue()));

        final long expectedNout = 2L*7L;
        final MutableInt cnt = new MutableInt(0);

        final ComputationGraphConfiguration.GraphBuilder graphBuilder = new NeuralNetConfiguration.Builder().graphBuilder()
                .addInputs("input")
                .setInputTypes(InputType.convolutional(9,9, 5))
                .addLayer("first", new Convolution2D.Builder(2,2).nOut(5).build(), "input");
        LayerBlockConfig.BlockInfo output = denseFunction.apply(2*3*7L).addLayers(new LayerSpyAdapter((layerName, layer, layerInputs) -> {
            assertEquals("Incorrect nOut!", expectedNout, ((FeedForwardLayer)layer).getNOut());
            cnt.increment();
        }, new GraphAdapter(graphBuilder)), new LayerBlockConfig.SimpleBlockInfo.Builder().setInputs(new String[] {"first"}).build());

        assertEquals("Incorrect number of blocks in stack!", 3, cnt.intValue());

        assertEquals("Incorrect nOut!", 2*3*7L, GraphBuilderUtil.getInputSize(
                output.getInputsNames()[0],
                graphBuilder));
    }
}