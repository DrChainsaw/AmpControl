package ampcontrol.model.training.model.layerblocks.graph;

import ampcontrol.model.training.model.layerblocks.Conv2DBatchNormAfter;
import ampcontrol.model.training.model.layerblocks.LayerBlockConfig;
import ampcontrol.model.training.model.layerblocks.adapters.BuilderAdapter;
import ampcontrol.model.training.model.layerblocks.adapters.GraphBuilderAdapter;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Dense stack. The given {@link LayerBlockConfig} will be stacked a given number of times. The input to the DenseStack
 * as well as all outputs from {@link LayerBlockConfig} l < L will be input to {@link LayerBlockConfig} L For each
 *{@link LayerBlockConfig} L <= nrofStacks.
 * <br><br>
 * See https://arxiv.org/abs/1608.06993
 *
 * @author Christian Skärby
 */
public class DenseStack implements LayerBlockConfig {

    private static final Logger log = LoggerFactory.getLogger(DenseStack.class);
    
    private LayerBlockConfig blockToStack = new Conv2DBatchNormAfter();
    private int nrofStacks = 3;
    private String namePrefix = "ds_";

    @Override
    public String name() {
        return nrofStacks + namePrefix + blockToStack.name();
    }

    @Override
    public BlockInfo addLayers(BuilderAdapter builder, BlockInfo info) {
        throw new IllegalArgumentException("Can only stack graphs!");
    }

    @Override
    public BlockInfo addLayers(GraphBuilderAdapter graphBuilder, BlockInfo info) {

        log.info("Create dense block: " + info);
        List<String> mergeVertexInputs = new ArrayList<>();

        mergeVertexInputs.addAll(Arrays.asList(info.getInputsNames()));
        int nrofOutputs = info.getPrevNrofOutputs();
        BlockInfo nextInfo = info;
        for (int i = 0; i < nrofStacks; i++) {

            log.info("Create dense sublayer: " + nextInfo);
            nextInfo = blockToStack.addLayers(graphBuilder, nextInfo);
            nrofOutputs += nextInfo.getPrevNrofOutputs();
            mergeVertexInputs.addAll(Arrays.asList(nextInfo.getInputsNames()));

            if (mergeVertexInputs.size() > 1) {
                final int mergeLayerInd = nextInfo.getPrevLayerInd() + 1;
                final String mergeName = nextInfo.getName(String.valueOf(nextInfo.getPrevLayerInd() + 1));
                log.info("Create dense vertex: " + nextInfo.getName(String.valueOf(mergeLayerInd)) + " for " + mergeVertexInputs);
                graphBuilder.addVertex(mergeName,
                        new MergeVertex(),
                        mergeVertexInputs.toArray(new String[]{}));
                nextInfo = new SimpleBlockInfo.Builder(nextInfo)
                        .setPrevLayerInd(mergeLayerInd)
                        .setInputs(new String[]{mergeName})
                        .setPrevNrofOutputs(nrofOutputs)
                        .build();
            }
        }
        return nextInfo;
    }

    /**
     * Set the {@link LayerBlockConfig} which shall be stacked densely
     * @param blockToStack
     * @return The {@link DenseStack}
     */
    public DenseStack setBlockToStack(LayerBlockConfig blockToStack) {
        this.blockToStack = blockToStack;
        return this;
    }

    /**
     * Set the number of times blockToStack shall be stacked
     * @param nrofStacks
     * @return The {@link DenseStack}
     */
    public DenseStack setNrofStacks(int nrofStacks) {
        this.nrofStacks = nrofStacks;
        return this;
    }

    /**
     * Sets a name prefix for all layers in the dense block.
     * @param namePrefix
     * @return The {@link DenseStack}
     */
    public DenseStack setNamePrefix(String namePrefix) {
        this.namePrefix = namePrefix;
        return this;
    }
}
