package ampcontrol.model.training.model.description;

import ampcontrol.model.training.data.iterators.MiniEpochDataSetIterator;
import ampcontrol.model.training.listen.ActivationContribution;
import ampcontrol.model.training.listen.NanScoreWatcher;
import ampcontrol.model.training.model.*;
import ampcontrol.model.training.model.builder.BlockBuilder;
import ampcontrol.model.training.model.builder.DeserializingModelBuilder;
import ampcontrol.model.training.model.builder.ModelBuilder;
import ampcontrol.model.training.model.evolve.CachedPopulation;
import ampcontrol.model.training.model.evolve.EvolvingPopulation;
import ampcontrol.model.training.model.evolve.Population;
import ampcontrol.model.training.model.evolve.TransformPopulation;
import ampcontrol.model.training.model.evolve.fitness.AddListener;
import ampcontrol.model.training.model.evolve.fitness.AggPolicy;
import ampcontrol.model.training.model.evolve.fitness.ClearListeners;
import ampcontrol.model.training.model.evolve.fitness.FitnessPolicyTraining;
import ampcontrol.model.training.model.evolve.mutate.Mutation;
import ampcontrol.model.training.model.evolve.mutate.NoutMutation;
import ampcontrol.model.training.model.evolve.mutate.PersistentSet;
import ampcontrol.model.training.model.evolve.mutate.layer.*;
import ampcontrol.model.training.model.evolve.mutate.state.*;
import ampcontrol.model.training.model.evolve.selection.*;
import ampcontrol.model.training.model.evolve.transfer.ParameterTransfer;
import ampcontrol.model.training.model.layerblocks.*;
import ampcontrol.model.training.model.layerblocks.adapters.AddVertexGraphAdapter;
import ampcontrol.model.training.model.layerblocks.adapters.GraphBuilderAdapter;
import ampcontrol.model.training.model.layerblocks.adapters.LayerSpyAdapter;
import ampcontrol.model.training.model.layerblocks.graph.SpyBlock;
import ampcontrol.model.training.model.naming.AddSuffix;
import ampcontrol.model.training.model.naming.FileNamePolicy;
import ampcontrol.model.training.model.vertex.EpsilonSpyVertex;
import ampcontrol.model.training.schedule.MinLim;
import ampcontrol.model.training.schedule.Mul;
import ampcontrol.model.training.schedule.epoch.Exponential;
import ampcontrol.model.training.schedule.epoch.Offset;
import ampcontrol.model.training.schedule.epoch.SawTooth;
import ampcontrol.model.training.schedule.epoch.Step;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.Builder;
import lombok.Getter;
import org.apache.commons.lang.mutable.MutableLong;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.LayerVertex;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.jetbrains.annotations.NotNull;
import org.nd4j.jita.memory.CudaMemoryManager;
import org.nd4j.linalg.activations.impl.ActivationReLU;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.schedule.ISchedule;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.function.UnaryOperator;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * Description for simple 2D convolutional networs which will evolve during training to perform architecture search.
 *
 * @author Christian Skärby
 */
public final class MutatingConv2dFactory {

    private static final Logger log = LoggerFactory.getLogger(MutatingConv2dFactory.class);

    private final MiniEpochDataSetIterator trainIter;
    private final MiniEpochDataSetIterator evalIter;
    private final int[] inputShape;
    private final String namePrefix;
    private final FileNamePolicy modelFileNamePolicy;

    private final static class ActivationContributionComparator implements Consumer<INDArray>, Comparator<Integer> {

        private INDArray activationContribution = null;
        private final String wsName = "ActContribCompWs" + this.toString().split("@")[1];
        private final WorkspaceConfiguration workspaceConfig = WorkspaceConfiguration.builder()
                .policyAllocation(AllocationPolicy.STRICT)
                .policyLearning(LearningPolicy.FIRST_LOOP)
                .policyMirroring(MirroringPolicy.HOST_ONLY)
                .policyReset(ResetPolicy.ENDOFBUFFER_REACHED)
                .policySpill(SpillPolicy.REALLOCATE)
                .initialSize(0)
                //.overallocationLimit(20)
                .build();

        @Override
        public int compare(Integer elem1, Integer elem2) {
            if (elem1.equals(elem2)) {
                return 0;
            }

            return -Double.compare(
                    activationContribution.getDouble(elem1),
                    activationContribution.getDouble(elem2));
        }

        @Override
        public void accept(INDArray activationContribution) {
            // log.info("Got contrib: " + activationContribution);
            try (MemoryWorkspace wss = Nd4j.getWorkspaceManager().getAndActivateWorkspace(workspaceConfig, wsName)) {
                if (this.activationContribution == null) {
                    this.activationContribution = activationContribution.dup().migrate(false);
                }
                this.activationContribution.addi(activationContribution);
            }
        }

        @Override
        protected void finalize() throws Throwable {
            Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread(workspaceConfig, wsName).destroyWorkspace();
        }
    }

    @Getter
    @Builder
    private final static class MutationLayerState implements Consumer<String> {
        @Builder.Default
        final Set<String> mutateNout = new LinkedHashSet<>();
        @Builder.Default
        final Set<String> mutateKernelSize = new LinkedHashSet<>();
        @Builder.Default
        final Set<String> removeLayers = new LinkedHashSet<>();

        /**
         * Accept a layer name to remove
         *
         * @param layerToRemove Layer name to remove
         */
        @Override
        public void accept(String layerToRemove) {
            mutateNout.remove(layerToRemove);
            mutateKernelSize.remove(layerToRemove);
            removeLayers.remove(layerToRemove);
        }

        public MutationLayerState clone() {
            return builder()
                    .mutateNout(new LinkedHashSet<>(mutateNout))
                    .mutateKernelSize(new LinkedHashSet<>(mutateKernelSize))
                    .removeLayers(new LinkedHashSet<>(removeLayers))
                    .build();
        }
    }

    private final static class GraphSpyAppender implements UnaryOperator<GraphBuilderAdapter> {

        private final MutatingConv2dFactory.MutationLayerState state;

        private GraphSpyAppender(MutationLayerState state) {
            this.state = state;
        }

        @Override
        public GraphBuilderAdapter apply(GraphBuilderAdapter graphBuilderAdapter) {
            final LayerSpyAdapter.LayerSpy nOutSpy = (layerName, layer, layerInputs) -> {
                if (layer instanceof ConvolutionLayer) {
                    state.mutateNout.add(layerName);
                } else if (layer instanceof DenseLayer) {
                    state.mutateNout.add(layerName);
                }
            };

            final LayerSpyAdapter.LayerSpy kernelSizeSpy = (layerName, layer, layerInputs) -> {
                if (layer instanceof ConvolutionLayer) {
                    state.mutateKernelSize.add(layerName);
                }
            };

            final LayerSpyAdapter.LayerSpy removeVertexSpy = (layerName, layer, layerInputs) -> {
                if (!(layer instanceof OutputLayer)) {
                    state.removeLayers.add(layerName);
                }
            };

            return new AddVertexGraphAdapter(new EpsilonSpyVertex(), state.mutateNout::contains,
                    new LayerSpyAdapter(nOutSpy,
                            new LayerSpyAdapter(kernelSizeSpy,
                                    new LayerSpyAdapter(removeVertexSpy,
                                            graphBuilderAdapter))));

        }
    }

    public MutatingConv2dFactory(MiniEpochDataSetIterator trainIter, MiniEpochDataSetIterator evalIter, int[] inputShape, String namePrefix, FileNamePolicy modelFileNamePolicy) {
        this.trainIter = trainIter;
        this.evalIter = evalIter;
        this.inputShape = inputShape;
        this.namePrefix = namePrefix;
        this.modelFileNamePolicy = modelFileNamePolicy;
    }

    /**
     * Adds the ModelHandles defined by this class to the given list
     *
     * @param modelData list to add models to
     */
    public void addModelData(List<ModelHandle> modelData) {

        final Function<Integer, FileNamePolicy> modelNamePolicyFactory = candInd -> new AddSuffix(File.separator + candInd);
        final FileNamePolicy evolvingSuffix = new AddSuffix("_evolving_train");
        final ModelComparatorRegistry comparatorRegistry = new ModelComparatorRegistry();
        // TODO: Some other way to do this please...
        final Set<String> allNoutMutationLayers = new LinkedHashSet<>();

        final String removeName = "_mutRemoveLayers.json";
        final String nOutName = "_mutNOutLayers.json";
        final String kernelSizeName = "_mutKernelSizeLayers.json";

        // Create model population
        final List<EvolvingGraphAdapter> initialPopulation = new ArrayList<>();
        IntStream.range(0, 20).forEach(candInd -> {

            final FileNamePolicy candNamePolicy = modelFileNamePolicy
                    .compose(evolvingSuffix)
                    .andThen(modelNamePolicyFactory.apply(candInd));

            final MutationLayerState mutationLayerState = MutationLayerState.builder()
                    .build();


            final GraphSpyAppender graphSpyBuilder = new GraphSpyAppender(mutationLayerState);

            final ModelBuilder baseBuilder = createModelBuilder(graphSpyBuilder);

            final ModelBuilder builder = new DeserializingModelBuilder(
                    candNamePolicy,
                    baseBuilder);

            try {
                final ComputationGraph graph = builder.buildGraph(); // Will also populate mutation layers in case graph is new.
                final String baseName = candNamePolicy.toFileName(builder.name());
                final SharedSynchronizedState<MutationLayerState> initialState = new SharedSynchronizedState<>(MutationLayerState.builder()
                        .removeLayers(new PersistentSet<>(baseName + removeName, mutationLayerState.getRemoveLayers()).get())
                        .mutateNout(new PersistentSet<>(baseName + nOutName, mutationLayerState.getMutateNout()).get())
                        .mutateKernelSize(new PersistentSet<>(baseName + kernelSizeName, mutationLayerState.getMutateKernelSize()).get())
                        .build());

                final UnaryOperator<SharedSynchronizedState.View<MutationLayerState>> copyState = view -> {
                    allNoutMutationLayers.addAll(view.get().getMutateNout());
                    return view.update(view.get().clone()).copy();
                };

                allNoutMutationLayers.addAll(initialState.view().get().getMutateNout());
                final Random seedGenNout = new Random(candInd);
                final Random seedGenKs = new Random(-candInd);
                final Random seedGenGraphAdd = new Random(candInd + 100);
                final Random seedRemoveLayer = new Random(-candInd - 100);
                final MutationState<ComputationGraphConfiguration.GraphBuilder> mutation =
                        AggMutationState.<ComputationGraphConfiguration.GraphBuilder>builder()
                                .first(new NoMutationStateWapper<>(gb -> {
                                    new ConvType(inputShape).addLayers(gb, new LayerBlockConfig.SimpleBlockInfo.Builder().build());
                                    return gb;
                                }))
                                .andThen(new GenericMutationState<>(
                                        initialState.view(),
                                        copyState,
                                        (str, state) -> new ObjectMapper().writeValue(new File(str + removeName), state.get().getRemoveLayers()),
                                        state -> createRemoveLayersMutation(state.get().getRemoveLayers(), state.get(), seedRemoveLayer.nextInt())))
                                .andThen(new GenericMutationState<>(
                                        initialState.view(),
                                        copyState,
                                        (str, state) -> new ObjectMapper().writeValue(new File(str + nOutName), state.get().getMutateNout()),
                                        state -> createNoutMutation(state.get().getMutateNout(), seedGenNout.nextInt())))
                                .andThen(new GenericMutationState<>(
                                        initialState.view(),
                                        copyState,
                                        (str, state) -> new ObjectMapper().writeValue(new File(str + kernelSizeName), state.get().getMutateKernelSize()),
                                        state -> createKernelSizeMutation(state.get().getMutateKernelSize(), seedGenKs.nextInt())))
                                .andThen(new GenericMutationState<>(
                                        initialState.view(),
                                        copyState,
                                        (str, state) -> {/* Not needed? Maybe seed... */},
                                        state -> createGraphMutation(new GraphSpyAppender(state.get()), seedGenGraphAdd.nextInt())))

                                .build();

                final EvolvingGraphAdapter adapter = candInd == 0 || graph.getIterationCount() > 0 ?
                        new EvolvingGraphAdapter(graph, mutation,
                                graphToTransfer -> new ParameterTransfer(graphToTransfer,
                                        Objects.requireNonNull(comparatorRegistry.get(graphToTransfer)))) :
                        new EvolvingGraphAdapter(mutateGraph(mutation, graph), mutation,
                                graphToTransfer -> new ParameterTransfer(graphToTransfer,
                                        Objects.requireNonNull(comparatorRegistry.get(graphToTransfer))));

                initialPopulation.add(adapter);

            } catch (IOException e) {
                e.printStackTrace();
            }
        });
        // Its either this or catch an exception since everything but the CudaMemoryManager throws an exception
        if (Nd4j.getMemoryManager() instanceof CudaMemoryManager) {
            Nd4j.getMemoryManager().purgeCaches();
        }

        final Population<ModelHandle> population = createPopulation(allNoutMutationLayers, comparatorRegistry, initialPopulation);

        final FileNamePolicy referenceSuffix = new AddSuffix("_reference_train");
        final ModelBuilder baseBuilder = createModelBuilder(UnaryOperator.identity());
        modelData.add(new GenericModelHandle(trainIter, evalIter,
                new GraphModelAdapter(
                        new DeserializingModelBuilder(modelFileNamePolicy.compose(referenceSuffix),
                                baseBuilder).buildGraph()),
                referenceSuffix.toFileName(baseBuilder.name())));
        modelData.add(new ModelHandlePopulation(population, evolvingSuffix.toFileName(baseBuilder.name()), modelNamePolicyFactory));
    }

    @NotNull
    private Population<ModelHandle> createPopulation(
            Set<String> mutateNoutLayers,
            ModelComparatorRegistry comparatorRegistry,
            List<EvolvingGraphAdapter> initialPopulation) {
        final Random rng = new Random(666);
        final MutableLong nrofParams = new MutableLong(0);
        final Population<ModelHandle> population = new CachedPopulation<>(
                new TransformPopulation<>(adapter -> new GenericModelHandle(
                        trainIter,
                        evalIter,
                        adapter,
                        "cand"), // Do something about the name...
                        new EvolvingPopulation<>(
                                // The initial population
                                initialPopulation,

                                // Policy for computing fitness and as of now, do some cleanup and add some checks
                                // TODO: Separate out prepare and clean stuff from actual fitness policy or rename FitnessPolicy to CandidateCreationHook or something
                                AggPolicy.<EvolvingGraphAdapter>builder()
                                        // Not a fitness policy
                                        .first(new ClearListeners<>())
                                        // Kind of a fitness policy
                                        .second(new AddListener<>(fitnessConsumer -> new NanScoreWatcher(() -> fitnessConsumer.accept(Double.MAX_VALUE))))
                                        // Not a fitness policy
                                        .andThen((adapter, consumer) -> {
                                            final Set<String> existingLayers = new LinkedHashSet<>(mutateNoutLayers);
                                            existingLayers.retainAll(((ComputationGraph) adapter.asModel()).getConfiguration().getVertices().keySet());
                                            for (String layerName : existingLayers) {
                                                final ActivationContributionComparator comparator = new ActivationContributionComparator();
                                                adapter.asModel().addListeners(new ActivationContribution(layerName, comparator));
                                                comparatorRegistry.add(adapter.asModel(), layerName, 0, comparator); // For Conv
                                                comparatorRegistry.add(adapter.asModel(), layerName, 1, comparator); // For Dense
                                            }
                                            return adapter;
                                        })
                                        // This is the actual fitness policy
                                        .andThen(new FitnessPolicyTraining<>(107))
                                        // Not a fitness policy
                                        .andThen((adapter, fitcons) -> {
                                            nrofParams.add(adapter.asModel().numParams());
                                            return adapter;
                                        })
                                        .build(),

                                // Polícy for selecting candidates after fitness has been reported

                                CompoundFixedSelection.<EvolvingGraphAdapter>builder()
                                        .andThen(2, new EliteSelection<>())
                                        .andThen(initialPopulation.size() - 2,
                                                new EvolveSelection<>(
                                                        new RouletteSelection<EvolvingGraphAdapter>(rng::nextDouble)))
                                        .build()
                        )));

        population.onChangeCallback(() -> {
            log.info("Avg nrof params: " + (nrofParams.doubleValue() / initialPopulation.size()));
            nrofParams.setValue(0);
        });
        return population;
    }

    private Mutation<ComputationGraphConfiguration.GraphBuilder> createNoutMutation(final Set<String> mutationLayers, int seed) {
        final Random rng = new Random(seed);
        final Set<NoutMutation.NoutMutationDescription> nOutMutationSet = mutationLayers.stream()
                .map(str -> NoutMutation.NoutMutationDescription.builder()
                        .layerName(str)
                        .mutateNout(nOut -> (long) Math.max(4, nOut + Math.max(nOut, 10) * 0.5 * (rng.nextDouble() - 0.5)))
                        .build())
                .collect(Collectors.toSet());
        return new NoutMutation(
                () -> nOutMutationSet.stream().filter(str -> rng.nextDouble() < 0.1));
    }

    private Mutation<ComputationGraphConfiguration.GraphBuilder> createKernelSizeMutation(
            final Set<String> mutationLayers,
            int seed) {
        final Random rng = new Random(seed);
        return new LayerContainedMutation(
                () -> mutationLayers.stream().filter(str -> rng.nextDouble() < 0.1)
                        .map(layerName ->
                                LayerContainedMutation.LayerMutation.builder()
                                        .mutationInfo(
                                                LayerMutationInfo.builder()
                                                        .layerName(layerName)
                                                        .build())
                                        .mutation(layerConfOpt -> Optional.ofNullable(layerConfOpt)
                                                .map(Layer::clone)
                                                .filter(layerConf -> layerConf instanceof ConvolutionLayer)
                                                .map(layerConf -> (ConvolutionLayer) layerConf)
                                                .map(convConf -> {
                                                    convConf.setKernelSize(IntStream.of(convConf.getKernelSize())
                                                            .map(orgSize -> Math.min(10, Math.max(1, orgSize + 1 - rng.nextInt(3))))
                                                            .toArray());
                                                    return convConf;
                                                })
                                                .orElseThrow(() -> new IllegalArgumentException("Could not mutate layer " + layerName + " from " + layerConfOpt)))
                                        .build())
        );
    }

    private Mutation<ComputationGraphConfiguration.GraphBuilder> createGraphMutation(
            UnaryOperator<GraphBuilderAdapter> spyFactory,
            int seed) {
        final Function<LayerBlockConfig, LayerBlockConfig> spyConfig = lbc -> new SpyBlock(lbc)
                .setFactory(spyFactory);

        final Random rng = new Random(seed);
        final List<Function<Long, LayerBlockConfig>> beforeGpBlocks = Arrays.asList(
                nOut -> new Conv2DBatchNormAfter()
                        .setConvolutionMode(ConvolutionMode.Same)
                        .setNrofKernels(nOut.intValue())
                        .setKernelSize_w(1 + rng.nextInt(10)/2)
                        .setKernelSize_h(1 + rng.nextInt(10)/2),
                nOut -> new Conv2DBatchNormBefore()
                        .setConvolutionMode(ConvolutionMode.Same)
                        .setNrofKernels(nOut.intValue())
                        .setKernelSize_w(1 + rng.nextInt(10)/2)
                        .setKernelSize_h(1 + rng.nextInt(10)/2),
                nOut -> new Conv2DBatchNormBetween()
                        .setConvolutionMode(ConvolutionMode.Same)
                        .setNrofKernels(nOut.intValue())
                        .setKernelSize_w(1 + rng.nextInt(10)/2)
                        .setKernelSize_h(1 + rng.nextInt(10)/2));

        final List<Function<Long, LayerBlockConfig>> afterGpBlocks = Arrays.asList(
                nOut -> new Dense().setHiddenWidth(nOut.intValue()));

        Function<Long, LayerBlockConfig> lbcBeforeGpSupplier = nOut -> beforeGpBlocks.get(rng.nextInt(beforeGpBlocks.size())).andThen(spyConfig).apply(nOut);
        Function<Long, LayerBlockConfig> lbcAfterGpSupplier = nOut -> afterGpBlocks.get(rng.nextInt(afterGpBlocks.size())).andThen(spyConfig).apply(nOut);

        final String afterGpStr = "mutafterGp_";
        return new GraphMutation(() -> Stream.of(GraphMutation.GraphMutationDescription.builder()
                .mutation(graphBuilder -> {
                    final Map<String, List<String>> validVertexes = graphBuilder.getVertexInputs().entrySet().stream()
                            // Skip spy vertexes as this will break Activation contribution
                            .filter(vertexInfo -> !vertexInfo.getKey().matches("^spy.*"))
                            // Skip vertexes which are not input to any other vertex (i.e they are output layers)
                            .filter(vertexInfo -> graphBuilder.getVertexInputs().values().stream()
                                    .flatMap(Collection::stream)
                                    .anyMatch(input -> vertexInfo.getKey().equals(input)))
                            // Skip vertex which have no inputs (i.e they are input vertexes assuming such a thing exists).
                            .filter(vertexInfo -> !vertexInfo.getValue().isEmpty())
                            .collect(Collectors.toMap(
                                    Map.Entry::getKey,
                                    Map.Entry::getValue
                            ));

                    final String[] inputNames = validVertexes.values().stream()
                            .skip(rng.nextInt(validVertexes.size()))
                            .findFirst().orElseThrow(() -> new IllegalStateException("Could not find inputs!"))
                            .toArray(new String[0]);

                    // log.info("Insert layer after " + inputNames[0]);

                    return Stream.of(inputNames)
                            .filter(vertexName -> !vertexName.contains(afterGpStr))
                            .filter(vertexName -> isAfterGlobPool(vertexName, graphBuilder))
                            .map(layer -> new BlockMutationFunction(
                                    lbcAfterGpSupplier,
                                    inputNames,
                                    newName -> createUniqueVertexName(
                                            afterGpStr + String.join("_", inputNames) + "_" + newName,
                                            graphBuilder)))
                            .findAny()
                            .orElseGet(() -> new BlockMutationFunction(
                                    lbcBeforeGpSupplier,
                                    inputNames,
                                    newName -> createUniqueVertexName(
                                            "mutbeforeGp_" + String.join("_", inputNames) + "_" + newName,
                                            graphBuilder)))
                            .apply(graphBuilder);

                }).build()
        )
                .filter(mut -> rng.nextDouble() < 0.05));
    }

    private static boolean isAfterGlobPool(String vertexName, ComputationGraphConfiguration.GraphBuilder graphBuilder) {
        if (Stream.of(graphBuilder.getVertices().get(vertexName))
                .filter(vertex -> vertex instanceof LayerVertex)
                .map(vertex -> (LayerVertex) vertex)
                .map(vertex -> vertex.getLayerConf().getLayer())
                .anyMatch(layer -> layer instanceof DenseLayer || layer instanceof GlobalPoolingLayer)
        ) {
            // log.info("vertex " + vertexName + " is after glob pool");
            return true;
        }

        List<String> inputs = graphBuilder.getVertexInputs().get(vertexName);
        if (inputs == null || inputs.isEmpty()) {
            // Found input layer before finding globpool or denselayer
            //log.info("vertex " + vertexName + " is before glob pool");
            return false;
        }
        return inputs.stream().anyMatch(vertexInputName -> isAfterGlobPool(vertexInputName, graphBuilder));
    }

    private static String createUniqueVertexName(String wantedName, ComputationGraphConfiguration.GraphBuilder graphBuilder) {
        return graphBuilder.getVertices().keySet().stream()
                .filter(wantedName::equals)
                .map(name -> createUniqueVertexName("_" + name, graphBuilder))
                .findAny()
                .orElse(wantedName);
    }

    private Mutation<ComputationGraphConfiguration.GraphBuilder> createRemoveLayersMutation(
            Set<String> mutationLayers,
            Consumer<String> removeListener,
            int seed) {
        Random rng = new Random(seed);
        return new GraphMutation(() -> mutationLayers.stream()
                .filter(str -> rng.nextDouble() < 0.05 / mutationLayers.size())
                .peek(vertexToRemove -> log.info("Attempt to remove " + vertexToRemove))
                // Collect subset to avoid that the removeListener causes ConcurrentModificationExceptions
                .collect(Collectors.toSet()).stream()
                .map(vertexToRemove -> GraphMutation.GraphMutationDescription.builder()
                        .mutation(graphBuilder -> {

                            if (graphBuilder.getVertexInputs().get(vertexToRemove).stream()
                                    .anyMatch(inputName -> graphBuilder.getNetworkInputs().contains(inputName))) {
                                return GraphMutation.InputsAndOutputNames.builder().build();
                            }

                            // Remove any spy vertices as well since they don't to anything useful by themselves
                            graphBuilder.getVertexInputs().entrySet().stream()
                                    .filter(vertexInfo -> vertexInfo.getValue().contains(vertexToRemove))
                                    .filter(vertexInfo -> vertexInfo.getKey().matches("^spy.*"))
                                    .map(Map.Entry::getKey)
                                    .collect(Collectors.toList())
                                    .forEach(spyVertexName ->
                                            new RemoveLayerFunction(spyVertexName).apply(graphBuilder)
                                    );
                            removeListener.accept(vertexToRemove);
                            return new RemoveLayerFunction(vertexToRemove).apply(graphBuilder);
                        })
                        .build()));
    }

    private ModelBuilder createModelBuilder(
            UnaryOperator<GraphBuilderAdapter> spyFactory) {

        final int schedPeriod = 50;
        final ISchedule lrSched = new Mul(new MinLim(0.02, new Step(4000, new Exponential(0.2))),
                new SawTooth(schedPeriod, 1e-6, 0.05));
        final ISchedule momSched = new Offset(schedPeriod / 2,
                new SawTooth(schedPeriod, 0.85, 0.95));

        final LayerBlockConfig pool = new Pool2D().setSize(2).setStride(2);
        return new BlockBuilder()
                .setUpdater(new Nesterovs(lrSched, momSched))
                .setNamePrefix(namePrefix)
                .first(new ConvType(inputShape))
                .andThen(new SpyBlock(new Conv2DBatchNormAfter()
                        .setKernelSize(3)
                        .setNrofKernels(24))
                        .setFactory(spyFactory))
                .andThen(pool)

                .andThen(new SpyBlock(new Conv2DBatchNormAfter()
                        .setKernelSize(2)
                        .setNrofKernels(49))
                        .setFactory(spyFactory))
                .andThen(pool)

                .andThen(new SpyBlock(new Conv2DBatchNormAfter()
                        .setKernelSize_h(1)
                        .setKernelSize_w(4)
                        .setNrofKernels(48))
                        .setFactory(spyFactory))
                .andThen(pool)

                .andThen(new GlobPool())
                .andThenStack(2)
                .of(new SpyBlock(new Dense()
                        .setHiddenWidth(75)
                        .setActivation(new ActivationReLU()))
                        .setFactory(spyFactory))
                .andFinally(new CenterLossOutput(trainIter.totalOutcomes())
                        .setAlpha(0.6)
                        .setLambda(0.003));
    }

    @NotNull
    private static ComputationGraph mutateGraph(Mutation<ComputationGraphConfiguration.GraphBuilder> mutation, ComputationGraph graph) {
        final ComputationGraph mutated = new ComputationGraph(mutation.mutate(
                new ComputationGraphConfiguration.GraphBuilder(graph.getConfiguration().clone(),
                        new NeuralNetConfiguration.Builder(graph.conf().clone())))
                .build());
        mutated.init();
        return mutated;
    }
}
