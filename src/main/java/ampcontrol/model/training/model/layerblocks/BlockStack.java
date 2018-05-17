package ampcontrol.model.training.model.layerblocks;


import ampcontrol.model.training.model.layerblocks.adapters.BuilderAdapter;
import ampcontrol.model.training.model.layerblocks.adapters.GraphBuilderAdapter;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;

/**
 * Applies the a {@link LayerBlockConfig} mulitple times, effectively creating a stack of its configuration.
 *
 * @author Christian Skärby
 */
public class BlockStack implements LayerBlockConfig {

    private int nrofStacks = 2;
    private LayerBlockConfig blockToStack = new Dense();


    @Override
    public String name() {
        return nrofStacks + "x_" + blockToStack.name();
    }

    @Override
    public BlockInfo addLayers(BuilderAdapter builder, BlockInfo info) {
       BlockInfo nextLayer = info;
        for(int i = 0; i < nrofStacks; i++) {
           nextLayer= blockToStack.addLayers(builder, nextLayer);
       }
       return nextLayer;
    }

    @Override
    public BlockInfo addLayers(GraphBuilderAdapter graphBuilder, BlockInfo info) {
        BlockInfo nextLayer = info;
        for(int i = 0; i < nrofStacks; i++) {
            nextLayer= blockToStack.addLayers(graphBuilder, nextLayer);
        }
        return nextLayer;
    }

    @Override
    public BlockInfo addLayers(ComputationGraphConfiguration.GraphBuilder graphBuilder, BlockInfo info) {
        BlockInfo nextLayer = info;
        for(int i = 0; i < nrofStacks; i++) {
            nextLayer= blockToStack.addLayers(graphBuilder, nextLayer);
        }
        return nextLayer;
    }

    /**
     * Sets the number of times the configuration shall be stacked
     * @param nrofStacks the number of times the configuration shall be stacked
     * @return the {@link BlockStack}
     */
    public BlockStack setNrofStacks(int nrofStacks) {
        this.nrofStacks = nrofStacks; return this;
    }

    /**
     * Sets the {@link LayerBlockConfig} which describes the configuration to be stacked.
     * @param blockToStack the {@link LayerBlockConfig} which describes the configuration to be stacked.
     * @return The blockStack
     */
    public BlockStack setBlockToStack(LayerBlockConfig blockToStack) {
        this.blockToStack = blockToStack; return this;
    }
}
