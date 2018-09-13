package ampcontrol.model.training.model.builder;

import ampcontrol.model.training.model.layerblocks.AggBlock;
import ampcontrol.model.training.model.layerblocks.BlockStack;
import ampcontrol.model.training.model.layerblocks.LayerBlockConfig;
import ampcontrol.model.training.model.layerblocks.graph.DenseStack;
import ampcontrol.model.training.model.layerblocks.graph.ForkAgg;
import ampcontrol.model.training.model.layerblocks.graph.MultiLevelAgg;
import ampcontrol.model.training.model.layerblocks.graph.ResBlock;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration.ListBuilder;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.schedule.ISchedule;
import org.nd4j.linalg.schedule.ScheduleType;
import org.nd4j.linalg.schedule.StepSchedule;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Wraps a {@link NeuralNetConfiguration} or a {@link ComputationGraphConfiguration} in a declarative API using
 * {@link LayerBlockConfig LayerBlockConfigs} which is probably only comprehensible to the author. Typical example
 * of coding to pass time while waiting to see if a model seems to converge in training.
 * <p>
 * Example of a simple resnet:
 * <p>
 * <pre>
 *               // Zero padding block which maintains output size after 4x4 convolution
 *               final LayerBlockConfig zeroPad4x4 = new ZeroPad()
 *                    .setPad_h_top(1)
 *                    .setPad_h_bot(2)
 *                    .setPad_w_left(1)
 *                    .setPad_w_right(2);
 *
 *               BlockBuilder bBuilder=new BlockBuilder()
 *                        .setNamePrefix(prefix)              // Set a name prefix for the model to e.g. describe preprocessing steps
 *                        .first(new ConvType(inputShape))    // First declare input type (does not have to be first block)
 *                        .andThen(zeroPad4x4)                // Begin by zero padding the input
 *                        .andThen(new Conv2DBatchNormAfter() // Then 4x4 convolution with batch normalization after activation
 *                          .setKernelSize(4)
 *                          .setNrofKernels(128))
 *                         .andThen(zeroPad4x4)               // Then zero pad again
 *                        .andThenStack(3)                    // Then repeat the following block 3 times
 *                        .aggRes()                           // Block to be repeated: first a residual block followed by more stuff until andFinally
 *                        .aggOf(new Conv2DBatchNormAfter()   // Each residual block is composed of all following blocks until andFinally
 *                          .setKernelSize(4)
 *                          .setNrofKernels(128))
 *                        .andFinally(zeroPad4x4)             // End each residual block with zero padding to maintain size
 *                        .andFinally(new Pool2D()            // End each of the 3 repeated blocks with max pooling
 *                          .setSize(2)
 *                          .setStride(2))
 *                        // .andThen(new GlobMeanMax())      // Uncomment to add Global mean max pooling
 *                        .andThenStack(2)                    // Then repeat the following block 2 times
 *                        .of(new Dense()                     // Block to be repeated: A fully connected layer with SELU
 *                          .setActivation(new ActivationSELU()))
 *                        .andThen(new DropOut()              // Add dropout just before the output
 *                          .setDropProb(dropOutProb))
 *                        .andFinally(new Output(nLabels));   // End the whole thing with an output layer
 * </pre>
 * Graph of result:
 * <pre>
 *
 *     Input
 *       |
 *    ZeroPad
 *       |
 *    Conv4x4
 *       |
 *    ZeroPad
 *       | \
 *       |  Conv4x4
 *       |  |
 *       |  ZeroPad
 *       | /
 *       +
 *       |
 *    MaxPool2x2
 *       | \
 *       |  Conv4x4
 *       |  |
 *       |  ZeroPad
 *       | /
 *       +
 *       |
 *    MaxPool2x2
 *       | \
 *       |  Conv4x4
 *       |  |
 *       |  ZeroPad
 *       | /
 *       +
 *       |
 *    MaxPool2x2
 *       |
 *     Dense
 *       |
 *     Dense
 *       |
 *    Dropout
 *       |
 *     Output
 * </pre>
 *
 * @author Christian Skärby
 */

public class BlockBuilder implements ModelBuilder {

    private static final Logger log = LoggerFactory.getLogger(BlockBuilder.class);

    private AggBlock layerBlockConfig;

    private String namePrefix = "";
    private int seed = 666;
    private WorkspaceMode trainWs = WorkspaceMode.ENABLED;
    private WorkspaceMode evalWs = WorkspaceMode.ENABLED;

    private ISchedule learningRateSchedule = new StepSchedule(ScheduleType.ITERATION, 0.05, 0.1, 40000);
    private IUpdater updater = new Adam(learningRateSchedule);

    @Override
    public MultiLayerNetwork build() {
        log.info("Creating model: " + name());

        final NeuralNetConfiguration.Builder builder = initBuilder();

        ListBuilder listBuilder = builder.list();
        LayerBlockConfig.BlockInfo info = new LayerBlockConfig.SimpleBlockInfo.Builder()
                .setPrevLayerInd(-1)
                .build();
        layerBlockConfig.addLayers(listBuilder, info);

        MultiLayerNetwork model = new MultiLayerNetwork(listBuilder.build());
        model.init();
        return model;
    }

    @Override
    public ComputationGraph buildGraph() {
        log.info("Creating graph: " + name());

        final NeuralNetConfiguration.Builder builder = initBuilder();

        ComputationGraphConfiguration.GraphBuilder graphBuilder = builder.graphBuilder();
        final String inputName = "input";
        graphBuilder.addInputs(inputName);
        LayerBlockConfig.BlockInfo info = new LayerBlockConfig.SimpleBlockInfo.Builder()
                .setPrevLayerInd(-1)
                .setInputs(new String[]{inputName})
                .build();
        final LayerBlockConfig.BlockInfo output = layerBlockConfig.addLayers(graphBuilder, info);
        graphBuilder.setOutputs(output.getInputsNames());

        ComputationGraph model = new ComputationGraph(graphBuilder.build());
        model.init();
        return model;
    }

    private NeuralNetConfiguration.Builder initBuilder() {
        return new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.RELU_UNIFORM)
                .activation(Activation.IDENTITY) // Will be set later on
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(updater)
                .trainingWorkspaceMode(trainWs)
                .inferenceWorkspaceMode(evalWs)
                .cacheMode(CacheMode.DEVICE);
    }


    /**
     * Returns the name of the model the builder will create.
     *
     * @return the name of the model
     */
    @Override
    public String name() {
        String updater = this.updater.getClass().getSimpleName();
        return namePrefix + layerBlockConfig.name() + updater;
    }

    /**
     * Sets a prefix for the model name e.g. to indicate what the input looks like
     *
     * @param prefix
     * @return the {@link BlockBuilder}
     */
    public BlockBuilder setNamePrefix(String prefix) {
        namePrefix = prefix;
        return this;
    }

    /**
     * Sets the seed
     *
     * @param seed
     * @return the {@link BlockBuilder}
     */
    public BlockBuilder setSeed(int seed) {
        this.seed = seed;
        return this;
    }

    /**
     * Sets the updater to use
     *
     * @param updater
     * @return the {@link BlockBuilder}
     */
    public BlockBuilder setUpdater(IUpdater updater) {
        this.updater = updater;
        return this;
    }

    /**
     * Sets the {@link WorkspaceMode} for training
     *
     * @param trainWs
     * @return the {@link BlockBuilder}
     */
    public BlockBuilder setTrainWs(WorkspaceMode trainWs) {
        this.trainWs = trainWs;
        return this;
    }

    /**
     * Sets the {@link WorkspaceMode} for evaluation (inference)
     *
     * @param evalWs
     * @return the {@link BlockBuilder}
     */
    public BlockBuilder setEvalWs(WorkspaceMode evalWs) {
        this.evalWs = evalWs;
        return this;
    }

    /**
     * Sets the first {@link LayerBlockConfig} of the model
     *
     * @param first
     * @return a {@link RootBuilder}
     */
    public RootBuilder first(LayerBlockConfig first) {
        layerBlockConfig = new AggBlock(first, "_t_");
        return new RootBuilder(this, layerBlockConfig);
    }

    /**
     * Horrendous base builder for constructing models based on {@link LayerBlockConfig LayerBlockCongis} in a
     * declarative way. Returns the appropriate builder type depending on what the next step in the model is.
     * Idea is that API shall be used without the end user having to see what horrors happen to make the API fluent
     * and declarative.
     * <br><br>
     * Worst part of the design is probably that new types of composing {@link LayerBlockConfig LayerBlockConfigs} tend
     * to require that new methods are created in all builders.
     *
     * @param <T> Type of the parentBuilder
     * @param <V> Type of this builder
     */
    private static class LayerBlockBuilder<T, V> {

        private final T parentBuilder;
        private AggBlock blockConf;

        private LayerBlockBuilder(T bb, AggBlock lbc) {
            this.parentBuilder = bb;
            this.blockConf = lbc;
        }

        /**
         * Adds one {@link LayerBlockConfig} to be applied after the last added {@link LayerBlockConfig}
         *
         * @param then the {@link LayerBlockConfig} to be added
         * @return the Builder
         */
        public V andThen(LayerBlockConfig then) {
            AggBlock next = new AggBlock(then, "_t_");
            blockConf.andThen(next);
            blockConf = next;
            return (V) this;
        }

        /**
         * Adds a {@link ResBlock} to be applied after the last added {@link LayerBlockConfig}. Returns a
         * {@link ResBlockBuilder} which operates on the added {@link ResBlock} and returns this builder when ready.
         *
         * @return a {@link ResBlockBuilder}
         */
        public ResBlockBuilder<V> andThenRes() {
            ResBlock resBlock = new ResBlock();
            AggBlock next = new AggBlock(resBlock, "_t_");
            blockConf.andThen(next);
            blockConf = next;
            return new ResBlockBuilder<>((V) this, resBlock);
        }

        /**
         * Adds a {@link BlockStack} to be applied after the last added {@link LayerBlockConfig}. Returns a
         * {@link StackBuilder} which operates on the added {@link BlockStack} and returns this builder when ready.
         *
         * @param nrofStacks Number of times the {@link LayerBlockConfig} of the stack shall be repeated
         * @return a {@link StackBuilder}
         */
        public StackBuilder<V> andThenStack(int nrofStacks) {
            BlockStack stack = new BlockStack().setNrofStacks(nrofStacks);
            AggBlock next = new AggBlock(stack, "_t_");
            blockConf.andThen(next);
            blockConf = next;
            return new StackBuilder<>((V) this, stack);
        }

        /**
         * Adds a {@link DenseStack} to be applied after the last added {@link LayerBlockConfig}. Returns a
         * {@link DenseStackBuilder} which operates on the added {@link DenseStack} and returns this builder when ready.
         *
         * @param nrofStacks Number of times the {@link LayerBlockConfig} of the stack shall be repeated
         * @return a {@link DenseStackBuilder}
         */
        public DenseStackBuilder<V> andThenDenseStack(int nrofStacks) {
            DenseStack stack = new DenseStack().setNrofStacks(nrofStacks);
            AggBlock next = new AggBlock(stack, "_t_");
            blockConf.andThen(next);
            blockConf = next;
            return new DenseStackBuilder<>((V) this, stack);
        }

        /**
         * Adds a {@link MultiLevelAgg} to be applied after the last added {@link LayerBlockConfig}. Returns a
         * {@link MultiLevelBuilder} which operates on the added {@link MultiLevelAgg} and returns this builder when
         * ready.
         *
         * @return a {@link MultiLevelBuilder}
         */
        public MultiLevelBuilder<V> multiLevel() {
            MultiLevelAgg multi = new MultiLevelAgg();
            AggBlock next = new AggBlock(multi);
            blockConf.andThen(next);
            blockConf = next;
            return new MultiLevelBuilder<>((V) this, multi);
        }

        /**
         * Adds a {@link ForkAgg} to be applied after the last added {@link LayerBlockBuilder}. Returns a
         * {@link ForkBuilder} which operates on the added {@link ForkAgg} and returns this builders parent builder when
         * ready.
         *
         * @return a {@link ForkBuilder}
         */
        public ForkBuilder<V> andThenFork() {
            final ForkAgg fork = new ForkAgg("_f_");
            AggBlock next = new AggBlock(fork);
            blockConf.andThen(next);
            blockConf = next;
            return new ForkBuilder<>((V) this, fork);
        }

        /**
         * Adds a last {@link LayerBlockConfig} and returns the parent builder
         *
         * @param last the {@link LayerBlockConfig} to be added
         * @return the parent builder
         */
        public T andFinally(LayerBlockConfig last) {
            blockConf.andThen(last);
            return parentBuilder;
        }
    }

    /**
     * Intermediate builder which configures a {@link BlockStack} and returns a parent builder when ready
     *
     * @param <T> The type of the parent builder
     */
    public static class StackBuilder<T> {
        private final BlockStack stackAgg;
        private final T parentBuilder;

        private StackBuilder(T lbb, BlockStack stackAgg) {
            this.stackAgg = stackAgg;
            this.parentBuilder = lbb;
        }

        /**
         * Sets the {@link LayerBlockConfig} to stack
         *
         * @param toStack the {@link LayerBlockConfig} to stack
         * @return the parent builder
         */
        public T of(LayerBlockConfig toStack) {
            stackAgg.setBlockToStack(toStack);
            return parentBuilder;
        }

        /**
         * Makes the stack a stack of {@link ResBlock ResBlocks}. Returns a {@link ResBlockBuilder} which operates on
         * the stacked {@link ResBlock} and returns this builders parent builder when ready.
         *
         * @return a {@link ResBlockBuilder}
         */
        public ResBlockBuilder<T> res() {
            ResBlock resBlock = new ResBlock();
            stackAgg.setBlockToStack(resBlock);
            return new ResBlockBuilder<>(parentBuilder, resBlock);
        }

        /**
         * Makes the stack a stack of an {@link AggBlock} with first {@link LayerBlockConfig} being a {@link ResBlock}.
         * Returns a {@link ResBlockBuilder} which operates on the {@link ResBlock} and returns a {@link NestBuilder}
         * which operates on the stacked {@link AggBlock} when ready. The {@link NestBuilder} in turn will return this
         * builders parent builder when ready.
         *
         * @return a {@link ResBlockBuilder}
         */
        public ResBlockBuilder<NestBuilder<T>> aggRes() {
            ResBlock resBlock = new ResBlock();
            AggBlock agg = new AggBlock(resBlock, "_w_");
            stackAgg.setBlockToStack(agg);
            return new ResBlockBuilder<>(new NestBuilder<>(parentBuilder, agg), resBlock);
        }

        /**
         * Makes the stack a stack of an {@link AggBlock} with the first {@link LayerBlockConfig} being the method
         * argument. Returns a {@link NestBuilder} which operates on the stacked {@link AggBlock} and returns this
         * builders parent builder when ready.
         *
         * @param toStack First {@link LayerBlockConfig} in the stacked {@link AggBlock}
         * @return a {@link NestBuilder}
         */
        public NestBuilder<T> aggOf(LayerBlockConfig toStack) {
            AggBlock toAgg = new AggBlock(toStack, "_w_");
            stackAgg.setBlockToStack(toAgg);
            return new NestBuilder<>(parentBuilder, toAgg);
        }

        /**
         * Makes the stack a stack of an {@link AggBlock} with first {@link LayerBlockConfig} being a new
         * {@link BlockStack}. Returns a {@link StackBuilder} which operates on the new {@link BlockStack} and returns a
         * {@link NestBuilder} which operates on the stacked {@link AggBlock} when ready. The {@link NestBuilder} in turn
         * will return this builders parent builder when ready.
         *
         * @param nrofStacks Number of times the {@link LayerBlockConfig} of the stack shall be repeated
         * @return a {@link StackBuilder}
         */
        public StackBuilder<NestBuilder<T>> aggStack(int nrofStacks) {
            BlockStack newStack = new BlockStack().setNrofStacks(nrofStacks);
            AggBlock agg = new AggBlock(newStack, "_w_");
            stackAgg.setBlockToStack(agg);
            return new StackBuilder<>(new NestBuilder<>(parentBuilder, agg), newStack);
        }

        /**
         * Makes the stack a stack of an {@link AggBlock} with first {@link LayerBlockConfig} being a {@link DenseStack}.
         * Returns a {@link DenseStackBuilder} which operates on the {@link DenseStack} and returns a {@link NestBuilder}
         * which operates on the stacked {@link AggBlock} when ready. The {@link NestBuilder} in turn will return this
         * builders parent builder when ready.
         *
         * @param nrofStacks Number of times the {@link LayerBlockConfig} of the stack shall be repeated
         * @return a {@link DenseStackBuilder}
         */
        public DenseStackBuilder<NestBuilder<T>> aggDenseStack(int nrofStacks) {
            DenseStack newStack = new DenseStack().setNrofStacks(nrofStacks);
            AggBlock agg = new AggBlock(newStack, "_w_");
            stackAgg.setBlockToStack(agg);
            return new DenseStackBuilder<>(new NestBuilder<>(parentBuilder, agg), newStack);
        }
    }

    /**
     * Intermediate builder which configures a {@link DenseStack} and returns a parent builder when ready
     *
     * @param <T> The type of the parent builder
     */
    public static class DenseStackBuilder<T> {
        private final DenseStack stackAgg;
        private final T parentBuilder;

        private DenseStackBuilder(T lbb, DenseStack stackAgg) {
            this.stackAgg = stackAgg;
            this.parentBuilder = lbb;
        }

        /**
         * Sets the {@link LayerBlockConfig} to stack
         *
         * @param toStack the {@link LayerBlockConfig} to stack
         * @return the parent builder
         */
        public T of(LayerBlockConfig toStack) {
            stackAgg.setBlockToStack(toStack);
            return parentBuilder;
        }

        /**
         * Makes the stack a stack of an {@link AggBlock} with the first {@link LayerBlockConfig} being the method
         * argument. Returns a {@link NestBuilder} which operates on the stacked {@link AggBlock} and returns this
         * builders parent builder when ready.
         *
         * @param toStack First {@link LayerBlockConfig} in the stacked {@link AggBlock}
         * @return a {@link NestBuilder}
         */
        public NestBuilder<T> aggOf(LayerBlockConfig toStack) {
            AggBlock toAgg = new AggBlock(toStack, "_w_");
            stackAgg.setBlockToStack(toAgg);
            return new NestBuilder<>(parentBuilder, toAgg);
        }

        /**
         * Makes the stack a stack of an {@link AggBlock} with first {@link LayerBlockConfig} being a {@link BlockStack}.
         * Returns a {@link StackBuilder} which operates on the {@link BlockStack} and returns a {@link NestBuilder}
         * which operates on the stacked {@link AggBlock} when ready. The {@link NestBuilder} in turn will return this
         * builders parent builder when ready.
         *
         * @param nrofStacks Number of times the {@link LayerBlockConfig} of the stack shall be repeated
         * @return a {@link StackBuilder}
         */
        public StackBuilder<NestBuilder<T>> aggStack(int nrofStacks) {
            BlockStack newStack = new BlockStack().setNrofStacks(nrofStacks);
            AggBlock agg = new AggBlock(newStack, "_w_");
            stackAgg.setBlockToStack(agg);
            return new StackBuilder<>(new NestBuilder<>(parentBuilder, agg), newStack);
        }
    }

    /**
     * Intermediate builder which configures a {@link ResBlock} and returns a parent builder when ready
     *
     * @param <T> The type of the parent builder
     */
    public static class ResBlockBuilder<T> {

        private final T parentBuilder;
        private final ResBlock blockConf;

        private ResBlockBuilder(T bb, ResBlock blockConf) {
            this.parentBuilder = bb;
            this.blockConf = blockConf;
        }

        /**
         * Sets the {@link LayerBlockConfig} which shall act as a residual block
         *
         * @param blockToRes the {@link LayerBlockConfig} which shall act as a residual block
         * @return the parent builder
         */
        public T of(LayerBlockConfig blockToRes) {
            blockConf.setBlockConfig(blockToRes);
            return parentBuilder;
        }

        /**
         * Makes the residual block to an {@link AggBlock} with the first {@link LayerBlockConfig} being the method
         * argument. Returns a {@link NestBuilder} which operates on the residual {@link AggBlock} and returns this
         * builders parent builder when ready.
         *
         * @param blockToRes First {@link LayerBlockConfig} in the residual {@link AggBlock}
         * @return a {@link NestBuilder}
         */
        public NestBuilder<T> aggOf(LayerBlockConfig blockToRes) {
            AggBlock next = new AggBlock(blockToRes, "_t_");
            blockConf.setBlockConfig(next);
            return new NestBuilder<>(parentBuilder, next);
        }

        /**
         * Makes the residual block to a {@link BlockStack} with the first {@link LayerBlockConfig} being the method
         * argument. Returns a {@link StackBuilder} which operates on the residual {@link BlockStack} and returns this
         * builders parent builder when ready.
         *
         * @param nrofStacks Number of times the {@link LayerBlockConfig} of the stack shall be repeated
         * @return a {@link NestBuilder}
         */
        public StackBuilder<T> ofStack(int nrofStacks) {
            BlockStack stack = new BlockStack().setNrofStacks(nrofStacks);
            blockConf.setBlockConfig(stack);
            return new StackBuilder<>(parentBuilder, stack);
        }

        /**
         * Makes the residual block to an {@link ForkAgg} Returns a {@link ForkBuilder} which operates on the residual
         * {@link ForkAgg} and returns this builders parent builder when ready.
         *
         * @return a {@link ForkBuilder}
         */
        public ForkBuilder<T> ofFork() {
            final ForkAgg next = new ForkAgg("_f_");
            blockConf.setBlockConfig(next);
            return new ForkBuilder<>(parentBuilder, next);
        }

        /**
         * Makes the residual block to an {@link AggBlock} with the first {@link LayerBlockConfig} being a
         * {@link ForkAgg}. Returns a {@link ForkBuilder} which operates on the {@link ForkAgg} and which in turn
         * returns the {@link NestBuilder} builder for the {@link AggBlock} when ready. The {@link NestBuilder} returns
         * the parent builder when ready.
         *
         * @return a {@link ForkBuilder}
         */
        public ForkBuilder<NestBuilder<T>> aggFork() {
            final ForkAgg fork = new ForkAgg("_f_");
            final AggBlock next = new AggBlock(fork);
            blockConf.setBlockConfig(next);
            return new ForkBuilder<>(new NestBuilder<>(parentBuilder, next), fork);
        }
    }

    /**
     * Intermediate builder which configures a {@link MultiLevelAgg} and returns a parent builder when ready
     *
     * @param <T> The type of the parent builder
     */
    public static class MultiLevelBuilder<T> {

        private final T parentBuilder;
        private MultiLevelAgg blockConf;

        private MultiLevelBuilder(T bb, MultiLevelAgg blockConf) {
            this.parentBuilder = bb;
            this.blockConf = blockConf;
        }

        /**
         * Adds one {@link LayerBlockConfig} to be applied after the last added {@link LayerBlockConfig}
         *
         * @param then the {@link LayerBlockConfig} to be added
         * @return the Builder
         */
        public MultiLevelBuilder<T> andThen(LayerBlockConfig then) {
            blockConf.andThen(then);
            return this;
        }

        /**
         * Adds one new {@link AggBlock} to be applied after the last added {@link LayerBlockConfig} with the method
         * argument as the first block. Returns a {@link NestBuilder} which configures the {@link AggBlock} and returns
         * this builder when ready.
         *
         * @param then the {@link LayerBlockConfig} to be added
         * @return a {@link NestBuilder}
         */
        public NestBuilder<MultiLevelBuilder<T>> andThenAgg(LayerBlockConfig then) {
            AggBlock next = new AggBlock(then, "_t_");
            blockConf.andThen(next);
            return new NestBuilder<>(this, next);
        }

        /**
         * Adds a {@link BlockStack} to be applied after the last added {@link LayerBlockConfig}. Returns a
         * {@link StackBuilder} which operates on the added {@link BlockStack} and returns this builder when ready.
         *
         * @param nrofStacks Number of times the {@link LayerBlockConfig} of the stack shall be repeated
         * @return a {@link StackBuilder}
         */
        public StackBuilder<MultiLevelBuilder<T>> andThenStack(int nrofStacks) {
            BlockStack stack = new BlockStack().setNrofStacks(nrofStacks);
            blockConf.andThen(stack);
            return new StackBuilder<>(this, stack);
        }

        /**
         * Adds a {@link ResBlock} to be applied after the last added {@link LayerBlockConfig}. Returns a
         * {@link ResBlockBuilder} which operates on the added {@link ResBlock} and returns this builder when ready.
         *
         * @return a {@link ResBlockBuilder}
         */
        public ResBlockBuilder<MultiLevelBuilder<T>> andThenRes() {
            ResBlock resBlock = new ResBlock();
            blockConf.andThen(resBlock);
            return new ResBlockBuilder<>(this, resBlock);
        }

        /**
         * Adds one new {@link AggBlock} to be applied after the last added {@link LayerBlockConfig} with a
         * {@link ResBlock} as the first block. Returns a {@link ResBlockBuilder} which operates on the {@link ResBlock}
         * and returns a {@link NestBuilder} which operates on the {@link AggBlock} when ready. The {@link NestBuilder}
         * in turn will return this builder when ready.
         *
         * @return a {@link NestBuilder}
         */
        public ResBlockBuilder<NestBuilder<MultiLevelBuilder<T>>> andThenAggRes() {
            ResBlock resBlock = new ResBlock();
            AggBlock agg = new AggBlock(resBlock);
            blockConf.andThen(agg);
            return new ResBlockBuilder<>(new NestBuilder<>(this, agg), resBlock);
        }

        /**
         * Returns the parent builder, effectively ending the multi-level feature aggregation.
         *
         * @return the parent builder
         */
        public T done() {
            return parentBuilder;
        }
    }

    /**
     * Intermediate builder which configures a {@link ForkAgg} and returns a parent builder when ready
     *
     * @param <T> The type of the parent builder
     */
    public static class ForkBuilder<T> {

        private final T parentBuilder;
        private ForkAgg blockConf;

        private ForkBuilder(T bb, ForkAgg blockConf) {
            this.parentBuilder = bb;
            this.blockConf = blockConf;
        }

        /**
         * Adds one {@link LayerBlockConfig} to be applied in parallel to the last added {@link LayerBlockConfig}
         *
         * @param then the {@link LayerBlockConfig} to be added
         * @return the Builder
         */
        public ForkBuilder<T> add(LayerBlockConfig then) {
            blockConf.add(then);
            return this;
        }

        /**
         * Adds one new {@link AggBlock} to be applied in parallel to the last added {@link LayerBlockConfig} with the method
         * argument as the first block. Returns a {@link NestBuilder} which configures the {@link AggBlock} and returns
         * this builder when ready.
         *
         * @param then the {@link LayerBlockConfig} to be added
         * @return a {@link NestBuilder}
         */
        public NestBuilder<ForkBuilder<T>> addAgg(LayerBlockConfig then) {
            AggBlock next = new AggBlock(then, "_t_");
            blockConf.add(next);
            return new NestBuilder<>(this, next);
        }

        /**
         * Adds a {@link BlockStack} to be applied in parallel to the last added {@link LayerBlockConfig}. Returns a
         * {@link StackBuilder} which operates on the added {@link BlockStack} and returns this builder when ready.
         *
         * @param nrofStacks Number of times the {@link LayerBlockConfig} of the stack shall be repeated
         * @return a {@link StackBuilder}
         */
        public StackBuilder<ForkBuilder<T>> addStack(int nrofStacks) {
            BlockStack stack = new BlockStack().setNrofStacks(nrofStacks);
            blockConf.add(stack);
            return new StackBuilder<>(this, stack);
        }

        /**
         * Adds a {@link ResBlock} to be applied in parallel to the last added {@link LayerBlockConfig}. Returns a
         * {@link ResBlockBuilder} which operates on the added {@link ResBlock} and returns this builder when ready.
         *
         * @return a {@link ResBlockBuilder}
         */
        public ResBlockBuilder<ForkBuilder<T>> addRes() {
            ResBlock resBlock = new ResBlock();
            blockConf.add(resBlock);
            return new ResBlockBuilder<>(this, resBlock);
        }

        /**
         * Adds one new {@link AggBlock} to be applied after the last added {@link LayerBlockConfig} with a
         * {@link ResBlock} as the first block. Returns a {@link ResBlockBuilder} which operates on the {@link ResBlock}
         * and returns a {@link NestBuilder} which operates on the {@link AggBlock} when ready. The {@link NestBuilder}
         * in turn will return this builder when ready.
         *
         * @return a {@link NestBuilder}
         */
        public ResBlockBuilder<NestBuilder<ForkBuilder<T>>> andThenAggRes() {
            ResBlock resBlock = new ResBlock();
            AggBlock agg = new AggBlock(resBlock);
            blockConf.add(agg);
            return new ResBlockBuilder<>(new NestBuilder<>(this, agg), resBlock);
        }

        /**
         * Returns the parent builder, effectively ending the fork.
         *
         * @return the parent builder
         */
        public T done() {
            return parentBuilder;
        }
    }

    /**
     * Root builder for adding {@link LayerBlockConfig LayerBlockConfigs} to a {@link BlockBuilder}.
     * {@see LayerBlockBuilder}
     */
    public static class RootBuilder extends LayerBlockBuilder<BlockBuilder, RootBuilder> {
        private RootBuilder(BlockBuilder bb, AggBlock lbc) {
            super(bb, lbc);
        }
    }

    /**
     * Nestled builder which adds {@link LayerBlockConfig LayerBlockConfigs} inside the context of some other builder.
     * {@see LayerBlockBuilder}
     *
     * @param <T> Type of the parent builder
     */
    public static class NestBuilder<T> extends LayerBlockBuilder<T, NestBuilder<T>> {
        private NestBuilder(T bb, AggBlock lbc) {
            super(bb, lbc);
        }
    }

}
