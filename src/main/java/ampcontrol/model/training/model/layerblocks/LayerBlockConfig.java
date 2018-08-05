package ampcontrol.model.training.model.layerblocks;

import ampcontrol.model.training.model.layerblocks.adapters.BuilderAdapter;
import ampcontrol.model.training.model.layerblocks.adapters.GraphBuilderAdapter;
import ampcontrol.model.training.model.layerblocks.adapters.ListAdapter;
import ampcontrol.model.training.model.layerblocks.graph.GraphBlockConfig;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.nd4j.linalg.activations.IActivation;

import java.util.Arrays;
import java.util.function.Function;

/**
 * Interface for a "Block" in a {@link NeuralNetConfiguration} or a {@link ComputationGraphConfiguration}.
 * Is expected to add layers or vertexes according to provided {@link BlockInfo} and return {@link BlockInfo} describing
 * the output. The name of the config is also provided.
 *
 * @author Christian Skärby
 */
public interface LayerBlockConfig extends GraphBlockConfig {

    /**
     * Name of the config as a string
     *
     * @return Name of the config as a string
     */
    String name();

    /**
     * Add layers to a {@link BuilderAdapter}.
     *
     * @param builder the builder
     * @param info    contains info on what the inputs are
     * @return {@link BlockInfo} with outputs
     */
    BlockInfo addLayers(BuilderAdapter builder, BlockInfo info);

    /**
     * Add layers to a {@link NeuralNetConfiguration.ListBuilder}.
     *
     * @param listBuilder the builder
     * @param info        contains info on what the inputs are
     * @return {@link BlockInfo} with outputs
     */
    default BlockInfo addLayers(NeuralNetConfiguration.ListBuilder listBuilder, BlockInfo info) {
        return addLayers(new ListAdapter(listBuilder), info);
    }

    @Override
    default BlockInfo addLayers(GraphBuilderAdapter graphBuilder, BlockInfo info) {
        return addLayers((BuilderAdapter) graphBuilder, info);
    }

    /**
     * Utility method to create a short name for an {@link IActivation} based on class.
     *
     * @param activation
     * @return short name of provided {@link IActivation}
     */
    static String actToStr(IActivation activation) {
        return activation.getClass().getSimpleName().replace("Activation", "").replace("Identity", "");
    }

    /**
     * Interface to send input information from one {@link LayerBlockConfig} to another.
     *
     * @author Christian Skärby
     */
    interface BlockInfo {

        /**
         * Returns index of previous layer
         *
         * @return index of previous layer
         */
        int getPrevLayerInd();

        /**
         * Returns number of outputs from the previous block
         *
         * @return number of outputs from the previous block
         */
        int getPrevNrofOutputs();

        /**
         * Creates a name based on a partial name. Typically used for adding prefix/suffix in nestled graphs to avoid
         * name aliasing.
         *
         * @param partName
         * @return a named based on a partial name.
         */
        String getName(String partName);

        /**
         * Returns the names of the input to the next block
         *
         * @return the names of the input to the next block
         */
        String[] getInputsNames();

    }

    /**
     * A simple buildable {@link BlockInfo}.
     *
     * @author Christian Skärby
     */
    class SimpleBlockInfo implements BlockInfo {

        private final int prevLayerInd;
        private final int prevNrofOutputs;
        private final Function<String, String> nameMapper;
        private final String[] inputs;

        /**
         * Constructor
         *
         * @param prevLayerInd    index of previous layer
         * @param prevNrofOutputs number of outputs from previous layer
         * @param nameMapper      maps a partial name to a "full" name
         * @param inputs          input names to a next layer
         */
        public SimpleBlockInfo(int prevLayerInd, int prevNrofOutputs, Function<String, String> nameMapper, String[] inputs) {
            this.prevLayerInd = prevLayerInd;
            this.prevNrofOutputs = prevNrofOutputs;
            this.nameMapper = nameMapper;
            this.inputs = inputs;
        }

        @Override
        public int getPrevLayerInd() {
            return prevLayerInd;
        }

        @Override
        public int getPrevNrofOutputs() {
            return prevNrofOutputs;
        }

        @Override
        public String getName(String partName) {
            return nameMapper.apply(partName);
        }

        @Override
        public String[] getInputsNames() {
            return inputs;
        }

        @Override
        public String toString() {
            return "SimpleBlockInfo{" +
                    "prevLayerInd=" + prevLayerInd +
                    ", inputs=" + Arrays.toString(inputs) +
                    '}';
        }

        /**
         * Builder for {@link SimpleBlockInfo}
         *
         * @author Christian Skärby
         */
        public static class Builder {

            private int prevLayerInd = 0;
            private int prevNrofOutputs = 0;
            private Function<String, String> nameMapper = Function.identity();
            private String[] inputs = new String[0];

            /**
             * Constructor
             */
            public Builder() {
            }

            /**
             * Constructor
             *
             * @param info default values will be set from this
             */
            public Builder(BlockInfo info) {
                prevLayerInd = info.getPrevLayerInd();
                prevNrofOutputs = info.getPrevNrofOutputs();
                nameMapper = str -> info.getName(str);
                inputs = info.getInputsNames();
            }

            /**
             * Builds a {@link SimpleBlockInfo}
             *
             * @return a {@link SimpleBlockInfo}
             */
            public SimpleBlockInfo build() {
                return new SimpleBlockInfo(prevLayerInd, prevNrofOutputs, nameMapper, inputs);
            }

            /**
             * Sets the prevLayerInd
             *
             * @param prevLayerInd
             * @return the {@link Builder}
             */
            public Builder setPrevLayerInd(int prevLayerInd) {
                this.prevLayerInd = prevLayerInd;
                return this;
            }

            /**
             * Sets the prevNrofOutputs
             *
             * @param prevNrofOutputs
             * @return the {@link Builder}
             */
            public Builder setPrevNrofOutputs(int prevNrofOutputs) {
                this.prevNrofOutputs = prevNrofOutputs;
                return this;
            }

            /**
             * Sets the nameMapper. Typically used for adding prefix/suffix to names in nestled graphs
             *
             * @param nameMapper
             * @return the {@link Builder}
             */
            public Builder setNameMapper(Function<String, String> nameMapper) {
                this.nameMapper = nameMapper;
                return this;
            }

            /**
             * Sets the inputs
             *
             * @param inputs
             * @return the {@link Builder}
             */
            public Builder setInputs(String[] inputs) {
                this.inputs = inputs;
                return this;
            }
        }
    }

}
