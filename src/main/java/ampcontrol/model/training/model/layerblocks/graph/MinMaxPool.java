package ampcontrol.model.training.model.layerblocks.graph;

import ampcontrol.model.training.model.layerblocks.LayerBlockConfig;
import ampcontrol.model.training.model.layerblocks.Pool2D;
import ampcontrol.model.training.model.layerblocks.adapters.BuilderAdapter;
import ampcontrol.model.training.model.layerblocks.adapters.GraphBuilderAdapter;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.graph.ScaleVertex;
import org.deeplearning4j.nn.conf.layers.PoolingType;

/**
 * Experimental (so far untested) design which concatenates the result of min and max pooling.
 * Obviously does not do anything useful with ReLU activation.
 *
 * @author Christian Skärby
 */
public class MinMaxPool implements LayerBlockConfig {
    
    private Pool2D poolBlock = new Pool2D();
    
    @Override
    public String name() {
        return "mm" + poolBlock.name();
    }

    @Override
    public BlockInfo addLayers(BuilderAdapter builder, BlockInfo info) {
        throw new IllegalArgumentException("Must work on graphBuilders!");
    }

    @Override
    public BlockInfo addLayers(GraphBuilderAdapter graphBuilder, BlockInfo info) {
        final BlockInfo maxPoolInfo = poolBlock.addLayers(graphBuilder, info);

        final String negName = info.getName("neg" + info.getPrevLayerInd());
        graphBuilder.addVertex(negName, new ScaleVertex(-1), info.getInputsNames());
        final BlockInfo negInfo = new SimpleBlockInfo.Builder(info)
                .setNameMapper(str -> "neg" + str)
                //.setPrevLayerInd(info.getPrevLayerInd()+1)
                .setInputs(new String [] {negName})
                .build();

        final BlockInfo minPoolInfo = poolBlock.addLayers(graphBuilder, negInfo);

        final String negNegName = info.getName("negNeg" + info.getPrevLayerInd());
        graphBuilder.addVertex(negNegName, new ScaleVertex(-1), minPoolInfo.getInputsNames());

        final String mergeName = info.getName("merge" + info.getPrevLayerInd());
        graphBuilder.addVertex(mergeName, new MergeVertex(),  maxPoolInfo.getInputsNames()[0], negNegName);

        return new SimpleBlockInfo.Builder(info)
                .setInputs(new String[] {mergeName})
                .setPrevNrofOutputs(minPoolInfo.getPrevNrofOutputs() + maxPoolInfo.getPrevNrofOutputs())
                .setPrevLayerInd(info.getPrevLayerInd()+1)
                .build();
    }

    /**
     * Sets both height and width of the pool kernel to the given size.
     *
     * @param size
     * @return the {@link MinMaxPool}
     */
    public MinMaxPool setSize(int size) {
         poolBlock.setSize(size); return this;
    }

    /**
     * Sets height of the pool kernel to the given size.
     * @param size_h
     * @return the {@link MinMaxPool}
     */
    public MinMaxPool setSize_h(int size_h) {
         poolBlock.setSize_h(size_h); return this;
    }
    /**
     * Sets width of the pool kernel to the given size.
     * @param size_w
     * @return the {@link MinMaxPool}
     */
    public MinMaxPool setSize_w(int size_w) {
         poolBlock.setSize_w(size_w); return this;
    }

    /**
     * Sets stride in both height and width to the given integer.
     * @param stride
     * @return the {@link MinMaxPool}
     */
    public MinMaxPool setStride(int stride) {
         poolBlock.setStride(stride); return this;
    }

    /**
     * Sets stride in height to the given integer.
     * @param stride_h
     * @return the {@link MinMaxPool}
     */
    public MinMaxPool setStride_h(int stride_h) {
         poolBlock.setStride_h(stride_h); return this;
    }

    /**
     * Sets stride in width to the given integer.
     * @param stride_w
     * @return the {@link MinMaxPool}
     */
    public MinMaxPool setStride_w(int stride_w) {
         poolBlock.setStride_w(stride_w); return this;
    }

    /**
     * Sets {@link PoolingType} of the underlying Pool2D.
     * @param type
     * @return the {@link MinMaxPool}
     */
    public MinMaxPool setType(PoolingType type) {
         poolBlock.setType(type); return this;
    }
}
