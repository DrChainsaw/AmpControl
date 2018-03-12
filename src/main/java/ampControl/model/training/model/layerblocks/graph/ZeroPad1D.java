package ampControl.model.training.model.layerblocks.graph;

import ampControl.model.training.model.layerblocks.LayerBlockConfig;
import ampControl.model.training.model.layerblocks.adapters.BuilderAdapter;
import ampControl.model.training.model.layerblocks.adapters.GraphBuilderAdapter;
import ampControl.model.training.model.vertex.ZeroPadding1DVertex;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


/**
 * Adds a {@link ZeroPadding1DVertex} to the graph.
 * 
 * @author Christian Skärby
 */
public class ZeroPad1D implements LayerBlockConfig {

    private static final Logger log = LoggerFactory.getLogger(SeBlock.class);

    private int paddingLeft = 0;
    private int paddingRight = 0;
    
    @Override
    public String name() {
        String padStr = paddingLeft == paddingRight ? ""+paddingLeft : paddingLeft + "_" + paddingRight;
        return "zp1_" + padStr;
    }

    @Override
    public BlockInfo addLayers(BuilderAdapter builder, BlockInfo info) {
        throw new RuntimeException("Can only do graphs!");
    }

    @Override
    public BlockInfo addLayers(GraphBuilderAdapter graphBuilder, BlockInfo info) {
        log.info("Zero pad 1D " + info);
        final int layerInd = info.getPrevLayerInd()+1;
        String thisLayer = info.getName(String.valueOf(layerInd));
        graphBuilder.addVertex(thisLayer, new ZeroPadding1DVertex(new int[] {paddingLeft, paddingRight}), info.getInputsNames());
        return new SimpleBlockInfo.Builder(info)
                .setPrevLayerInd(layerInd)
                .setInputs(new String[] {thisLayer})
                .build();
    }

    /**
     * Sets the amount of padding in both left and right direction
     * @param padding
     * @return the {@link ZeroPad1D}
     */
    public ZeroPad1D setPadding(int padding) {
        this.paddingLeft = padding;
        this.paddingRight = padding; return this;
    }

    /**
     * Sets the amount of padding in left direction
     * @param paddingLeft
     * @return the {@link ZeroPad1D}
     */
    public ZeroPad1D setPaddingLeft(int paddingLeft) {
        this.paddingLeft = paddingLeft; return this;
    }

    /**
     * Sets the amount of padding in right direction
     * @param paddingRight
     * @return the {@link ZeroPad1D}
     */
    public ZeroPad1D setPaddingRight(int paddingRight) {
        this.paddingRight = paddingRight; return this;
    }
}
