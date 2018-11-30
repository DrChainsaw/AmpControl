package ampcontrol.model.training.model.description;

import ampcontrol.model.training.data.iterators.MiniEpochDataSetIterator;
import ampcontrol.model.training.listen.NanScoreWatcher;
import ampcontrol.model.training.model.*;
import ampcontrol.model.training.model.builder.BlockBuilder;
import ampcontrol.model.training.model.builder.DeserializingModelBuilder;
import ampcontrol.model.training.model.builder.ModelBuilder;
import ampcontrol.model.training.model.builder.OverrideName;
import ampcontrol.model.training.model.evolve.CachedPopulation;
import ampcontrol.model.training.model.evolve.EvolvingPopulation;
import ampcontrol.model.training.model.evolve.Population;
import ampcontrol.model.training.model.evolve.TransformPopulation;
import ampcontrol.model.training.model.evolve.crossover.graph.GraphInfo;
import ampcontrol.model.training.model.evolve.crossover.graph.NoCrossover;
import ampcontrol.model.training.model.evolve.crossover.graph.SinglePoint;
import ampcontrol.model.training.model.evolve.crossover.state.CrossoverState;
import ampcontrol.model.training.model.evolve.crossover.state.GenericCrossoverState;
import ampcontrol.model.training.model.evolve.fitness.*;
import ampcontrol.model.training.model.evolve.mutate.*;
import ampcontrol.model.training.model.evolve.mutate.layer.*;
import ampcontrol.model.training.model.evolve.mutate.layer.blockfunctions.DenseStackFunction;
import ampcontrol.model.training.model.evolve.mutate.layer.blockfunctions.ForkFunction;
import ampcontrol.model.training.model.evolve.mutate.layer.blockfunctions.ListFunction;
import ampcontrol.model.training.model.evolve.mutate.state.AggMutationState;
import ampcontrol.model.training.model.evolve.mutate.state.GenericMutationState;
import ampcontrol.model.training.model.evolve.mutate.state.MutationState;
import ampcontrol.model.training.model.evolve.mutate.state.NoMutationStateWapper;
import ampcontrol.model.training.model.evolve.selection.*;
import ampcontrol.model.training.model.evolve.state.AccessibleState;
import ampcontrol.model.training.model.evolve.state.GenericState;
import ampcontrol.model.training.model.evolve.state.PersistentSet;
import ampcontrol.model.training.model.evolve.state.SharedSynchronizedState;
import ampcontrol.model.training.model.evolve.state.SharedSynchronizedState.View;
import ampcontrol.model.training.model.evolve.transfer.ParameterTransfer;
import ampcontrol.model.training.model.layerblocks.*;
import ampcontrol.model.training.model.layerblocks.adapters.AddVertexGraphAdapter;
import ampcontrol.model.training.model.layerblocks.adapters.GraphBuilderAdapter;
import ampcontrol.model.training.model.layerblocks.adapters.LayerSpyAdapter;
import ampcontrol.model.training.model.layerblocks.graph.ResBlock;
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
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration.GraphBuilder;
import org.deeplearning4j.nn.conf.graph.LayerVertex;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.jetbrains.annotations.NotNull;
import org.nd4j.jita.memory.CudaMemoryManager;
import org.nd4j.linalg.activations.impl.ActivationReLU;
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
 * Description for 2D convolutional networks which will evolve during training to perform some kind of architecture
 * search.
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


    @Getter
    @Builder
    private final static class MutationLayerState implements Consumer<String> {
        @Builder.Default
        final Set<String> mutateNout = new LinkedHashSet<>();
        @Builder.Default
        final Set<String> mutateKernelSize = new LinkedHashSet<>();
        @Builder.Default
        final Set<String> removeLayers = new LinkedHashSet<>();

        private final static String removeName = "_mutRemoveLayers.json";
        private final static String nOutName = "_mutNOutLayers.json";
        private final static String kernelSizeName = "_mutKernelSizeLayers.json";

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

        public void modify(Consumer<Set<String>> modification) {
            modification.accept(mutateNout);
            modification.accept(mutateKernelSize);
            modification.accept(removeLayers);
        }

        public void merge(MutationLayerState other) {
            mutateNout.addAll(other.mutateNout);
            mutateKernelSize.addAll(other.mutateKernelSize);
            removeLayers.addAll(other.removeLayers);
        }

        static MutationLayerState fromFile(String baseName, MutationLayerState mutationLayerState) throws IOException {
            return MutationLayerState.builder()
                    .removeLayers(new PersistentSet<>(baseName + removeName, mutationLayerState.getRemoveLayers()).get())
                    .mutateNout(new PersistentSet<>(baseName + nOutName, mutationLayerState.getMutateNout()).get())
                    .mutateKernelSize(new PersistentSet<>(baseName + kernelSizeName, mutationLayerState.getMutateKernelSize()).get())
                    .build();
        }

        public void save(String fileName) throws IOException {
            new ObjectMapper().writeValue(new File(fileName + removeName), removeLayers);
            new ObjectMapper().writeValue(new File(fileName + nOutName), mutateNout);
            new ObjectMapper().writeValue(new File(fileName + kernelSizeName), mutateKernelSize);
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

        // Create model population
        final List<EvolvingGraphAdapter<View<MutationLayerState>>> initialPopulation = new ArrayList<>();
        IntStream.range(0, 30).forEach(candInd -> {

            final FileNamePolicy candNamePolicy = modelFileNamePolicy
                    .compose(evolvingSuffix)
                    .andThen(modelNamePolicyFactory.apply(candInd));

            final MutationLayerState mutationLayerState = MutationLayerState.builder()
                    .build();


            final GraphSpyAppender graphSpyBuilder = new GraphSpyAppender(mutationLayerState);

            final ModelBuilder baseBuilder;
            if (candInd == 0) {
                baseBuilder = createSeedModelBuilder(graphSpyBuilder);
            } else {
                baseBuilder = new OverrideName(createSeedModelBuilder(UnaryOperator.identity()).name(),
                        createRandomModelBuilder(new Random(candInd + 666), graphSpyBuilder));
            }

            final ModelBuilder builder = new DeserializingModelBuilder(
                    candNamePolicy,
                    baseBuilder);

            final ComputationGraph graph = builder.buildGraph(); // Will also populate mutation layers in case graph is new.

            final String baseName = candNamePolicy.toFileName(builder.name());

            final EvolvingGraphAdapter<View<MutationLayerState>> adapter = EvolvingGraphAdapter.<View<MutationLayerState>>builder(graph)
                    .evolutionState(createInitialEvolutionState(mutationLayerState, baseName))
                    .mutation(createMutation(candInd))
                    .crossover(createCrossover(candInd))
                    .paramTransfer((nameToVertex, vertexToGraph) -> new ParameterTransfer(
                            nameToVertex,
                            // ParameterTransfer wants a mapping from vertex name to comparator. comparatorRegistry
                            // has one such mapping for each ComputationGraph, but we first need to determine
                            // which ComputationGraph is used for the given vertex
                            Objects.requireNonNull(
                                    vertexName -> comparatorRegistry.get(
                                            vertexToGraph.apply(nameToVertex.apply(vertexName))).apply(vertexName))
                    ))
                    .build();

            initialPopulation.add(adapter);

        });
        // Its either this or catch an exception since everything but the CudaMemoryManager throws an exception
        if (Nd4j.getMemoryManager() instanceof CudaMemoryManager) {
            Nd4j.getMemoryManager().purgeCaches();
        }

        final Population<ModelHandle> population = createPopulation(comparatorRegistry, initialPopulation);

        final FileNamePolicy referenceSuffix = new AddSuffix("_reference_train");
        final ModelBuilder baseBuilder = createSeedModelBuilder(UnaryOperator.identity());
        modelData.add(new GenericModelHandle(trainIter, evalIter,
                new GraphModelAdapter(
                        new DeserializingModelBuilder(modelFileNamePolicy.compose(referenceSuffix),
                                baseBuilder).buildGraph()),
                referenceSuffix.toFileName(baseBuilder.name())));
        modelData.add(new ModelHandlePopulation(population, evolvingSuffix.toFileName(baseBuilder.name()), modelNamePolicyFactory));
    }

    private MutationState<GraphBuilder, SharedSynchronizedState.View<MutationLayerState>> createMutation(
            int mutationBaseSeed) {

        final Random seedGenNout = new Random(mutationBaseSeed);
        final Random seedGenKs = new Random(-mutationBaseSeed);
        final Random seedGenGraphAdd = new Random(mutationBaseSeed + 100);
        final Random seedRemoveLayer = new Random(-mutationBaseSeed - 100);
        final Random memUsageRng = new Random(mutationBaseSeed + 1000);

        return AggMutationState.<GraphBuilder, View<MutationLayerState>>builder()
                .first(new NoMutationStateWapper<>(gb -> {
                    new ConvType(inputShape).addLayers(gb, new LayerBlockConfig.SimpleBlockInfo.Builder().build());
                    return gb;
                }))
                .andThen(new GenericMutationState<>(
                        state -> new MemoryAwareMutation<>(memUsage -> {
                            log.info("Create mutation from memusage: " + memUsage);
                            if (memUsage + 0.2 > memUsageRng.nextDouble()) {
                                // Use SuppliedMutation to be able to create mutations after RemoveLayersMutation has done its thing
                                // Why not remove last? Issue with size of weights when remove happens after nout mutation.
                                return AggMutation.<GraphBuilder>builder()
                                        .andThen(new SuppliedMutation<>(
                                                () -> createRemoveLayersMutation(state.get().getRemoveLayers(), state.get(), seedRemoveLayer.nextInt())))
                                        .andThen(new SuppliedMutation<>(
                                                () -> createNoutMutation(state.get().getMutateNout(), seedGenNout.nextInt(), 1)))
                                        .andThen(new SuppliedMutation<>(
                                                () -> createKernelSizeMutation(state.get().getMutateKernelSize(), seedGenKs.nextInt(), -1)))
                                        .build();
                            } else {
                                return AggMutation.<GraphBuilder>builder()
                                        .first(new SuppliedMutation<>(() ->
                                                createNoutMutation(state.get().getMutateNout(), seedGenNout.nextInt(), 0)))
                                        .andThen(new SuppliedMutation<>(() ->
                                                createKernelSizeMutation(state.get().getMutateKernelSize(), seedGenKs.nextInt(), 1)))
                                        .andThen(new SuppliedMutation<>(() ->
                                                createAddBlockMutation(new GraphSpyAppender(state.get()), seedGenGraphAdd.nextInt())))
                                        .build();
                            }
                        })
                ))
                .build();
    }

    private CrossoverState<GraphInfo, View<MutationLayerState>> createCrossover(int seed) {

        final Random rng = new Random(seed);

        return new GenericCrossoverState<>(
                (thisState, otherState, thisInput, otherInput, result) -> mergeMutationState(
                        thisState.get(),
                        otherState.get(),
                        thisInput,
                        otherInput,
                        result),
                state -> (bottom, top) -> {
                    new ConvType(inputShape).addLayers(bottom.builder(), new LayerBlockConfig.SimpleBlockInfo.Builder().build());
                    new ConvType(inputShape).addLayers(top.builder(), new LayerBlockConfig.SimpleBlockInfo.Builder().build());

                    if (rng.nextDouble() < 0.02) {
                        return new SinglePoint(() ->
                                new SinglePoint.PointSelection(
                                        Math.min(1d, Math.max(-1d, rng.nextGaussian() / 3)),
                                        rng.nextDouble())).cross(bottom, top);
                    }
                    return new NoCrossover().cross(bottom, top);
                });

    }

    private void mergeMutationState(
            MutationLayerState first,
            MutationLayerState second,
            GraphInfo firstInput,
            GraphInfo secondInput,
            GraphInfo result
    ) {
        first.modify(set -> filterSet(set, firstInput, result));
        second.modify(set -> filterSet(set, secondInput, result));
        first.merge(second);
    }

    private Stream<String> filterSet(
            Set<String> set,
            GraphInfo input,
            GraphInfo result) {
        return result.verticesFrom(input)
                .filter(nameMapping -> set.contains(nameMapping.getOldName()))
                .map(GraphInfo.NameMapping::getNewName);
    }

    private AccessibleState<View<MutationLayerState>> createInitialEvolutionState(MutationLayerState mutationLayerState, String baseName) {
        try {
            final SharedSynchronizedState<MutationLayerState> initialState =
                    new SharedSynchronizedState<>(MutationLayerState.fromFile(baseName, mutationLayerState));

            final UnaryOperator<View<MutationLayerState>> copyState =
                    view -> view.copy().update(view.get().clone());

            return new GenericState<>(initialState.view(),
                    copyState,
                    (str, state) -> state.get().save(str));
        } catch (IOException e) {
            throw new IllegalArgumentException("Failed to create mutation!", e);
        }
    }

    @NotNull
    private <S> Population<ModelHandle> createPopulation(
            ModelComparatorRegistry comparatorRegistry,
            List<EvolvingGraphAdapter<S>> initialPopulation) {

        final Random rng = new Random(666);
        final MutableLong nrofParams = new MutableLong(0);
        final Limit.FixedTotalLimit total = new Limit.FixedTotalLimit(initialPopulation.size());

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
                                AggPolicy.<EvolvingGraphAdapter<S>>builder()
                                        // Not a fitness policy
                                        .first(new ClearListeners<>())
                                        // Kind of a fitness policy
                                        .second(new AddListener<>(fitnessConsumer -> NanScoreWatcher.once(() -> fitnessConsumer.accept(Double.MAX_VALUE))))
                                        // Not a fitness policy
                                        .andThen(new InstrumentEpsilonSpies<>(comparatorRegistry))
                                        // This is the actual fitness policy
                                        .andThen(new FitnessPolicyTraining<>(107))
                                        // Not a fitness policy
                                        .andThen((adapter, fitcons) -> {
                                            nrofParams.add(adapter.asModel().numParams());
                                            return adapter;
                                        })
                                        .build(),

                                // Polícy for selecting candidates after fitness has been reported
                                CompoundSelection.<EvolvingGraphAdapter<S>>builder()
                                        .andThen(total.limit(2,
                                                new EliteSelection<>()))
                                        .andThen(total.limit(3,
                                                new CrossoverSelection<EvolvingGraphAdapter<S>>(
                                                        (cand, cands) -> {
                                                            final int selected = rng.nextInt(cands.size());
                                                            log.info("Selected crossover mate: " + selected);
                                                            return cands.get(selected);
                                                        },
                                                        new RouletteSelection<>(rng::nextDouble))))
                                        .andThen(total.last(
                                                new EvolveSelection<EvolvingGraphAdapter<S>>(
                                                        new RouletteSelection<>(rng::nextDouble))))
                                        .build()
                        )));

        population.onChangeCallback(() -> {
            log.info("Avg nrof params: " + (nrofParams.doubleValue() / initialPopulation.size()));
            nrofParams.setValue(0);
        });
        return population;
    }

    private Mutation<GraphBuilder> createNoutMutation(
            final Set<String> mutationLayers,
            int seed,
            double rngOffset) {
        final Random rng = new Random(seed);
        final Set<NoutMutation.NoutMutationDescription> nOutMutationSet = mutationLayers.stream()
                .map(str -> NoutMutation.NoutMutationDescription.builder()
                        .layerName(str)
                        .mutateNout(nOut -> (long) Math.max(4, nOut + Math.max(nOut, 10) * 0.1 * (rng.nextDouble() - rngOffset)))
                        .build())
                .collect(Collectors.toSet());
        return new NoutMutation(
                () -> nOutMutationSet.stream().filter(str -> rng.nextDouble() < 0.08));
    }

    private Mutation<GraphBuilder> createKernelSizeMutation(
            final Set<String> mutationLayers,
            int seed,
            int rngSign) {
        final Random rng = new Random(seed);
        return new LayerContainedMutation(
                () -> mutationLayers.stream().filter(str -> rng.nextDouble() < 0.03)
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
                                                            .map(orgSize -> Math.min(10, Math.max(1, orgSize + rngSign * rng.nextInt(2))))
                                                            .toArray());
                                                    return convConf;
                                                })
                                                .orElseThrow(() -> new IllegalArgumentException("Could not mutate layer " + layerName + " from " + layerConfOpt)))
                                        .build())
        );
    }

    private Mutation<GraphBuilder> createAddBlockMutation(
            UnaryOperator<GraphBuilderAdapter> spyFactory,
            int seed) {

        final Random rng = new Random(seed);

        final Function<Long, LayerBlockConfig> lbcBeforeGpSupplier = createBeforeGlobPoolLayerFactory(spyFactory, rng)
                .andThen(block -> rng.nextDouble() < 0.3 ? new ResBlock().setBlockConfig(block) : block);
        final Function<Long, LayerBlockConfig> lbcAfterGpSupplier = createAfterGlobPoolLayerFactory(spyFactory, rng);

        final String afterGpStr = "mutafterGp_";
        return new GraphMutation(() -> Stream.of(GraphMutation.GraphMutationDescription.builder()
                .mutation(graphBuilder -> {
                    final Map<String, List<String>> validVertexes = graphBuilder.getVertexInputs().entrySet().stream()
                            // Skip spy vertices as this will break Activation contribution
                            .filter(vertexInfo -> !vertexInfo.getKey().matches("^spy.*"))
                            // Skip vertices which are not input to any other vertex (i.e they are output layers)
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
                .filter(mut -> rng.nextDouble() < 0.1));
    }

    @NotNull
    private static Function<Long, LayerBlockConfig> createAfterGlobPoolLayerFactory(UnaryOperator<GraphBuilderAdapter> spyFactory, Random rng) {

        final Function<LayerBlockConfig, LayerBlockConfig> spyConfig = lbc -> new SpyBlock(lbc)
                .setFactory(spyFactory);

        final Function<Long, LayerBlockConfig> afterGpBlocks = ListFunction.builder()
                .function(nOut -> new Dense().setHiddenWidth(nOut.intValue()))
                .indexSupplier(rng::nextInt)
                .build();
        return nOut -> afterGpBlocks.andThen(spyConfig).apply(nOut);
    }

    @NotNull
    private static Function<Long, LayerBlockConfig> createBeforeGlobPoolLayerFactory(UnaryOperator<GraphBuilderAdapter> spyFactory, Random rng) {
        final Function<LayerBlockConfig, LayerBlockConfig> spyConfig = lbc -> new SpyBlock(lbc)
                .setFactory(spyFactory);

        final Function<Long, LayerBlockConfig> one = ListFunction.allConv2D(rng).build();

        // 2-5 paths
        final Function<Long, LayerBlockConfig> forkFunction = new ForkFunction(() -> rng.nextInt(3) + 2, one);

        // 10% chance of fork
        final Function<Long, LayerBlockConfig> maybeFork = nOut -> rng.nextDouble() < 0.1 ? forkFunction.apply(nOut) : one.apply(nOut);

        // Select one random possible dense stack size
        final Function<Long, LayerBlockConfig> denseFunction = new DenseStackFunction(stackChoices ->
                rng.ints(0, stackChoices.size())
                        .limit(Math.max(1,rng.nextInt(stackChoices.size() - 1)))
                        .mapToLong(stackChoices::get)
                        .reduce(1L, (l1, l2) -> l1 * l2), maybeFork);

        // 10% chance of dense stack
        final Function<Long, LayerBlockConfig> maybeDenseMaybeFork = nOut -> rng.nextDouble() < 0.1 ? denseFunction.apply(nOut) : maybeFork.apply(nOut);
        return maybeDenseMaybeFork.andThen(spyConfig);
    }

    private static boolean isAfterGlobPool(String vertexName, GraphBuilder graphBuilder) {
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

    private static String createUniqueVertexName(String wantedName, GraphBuilder graphBuilder) {
        return graphBuilder.getVertices().keySet().stream()
                .filter(wantedName::equals)
                .map(name -> createUniqueVertexName("_" + name, graphBuilder))
                .findAny()
                .orElse(wantedName);
    }

    private Mutation<GraphBuilder> createRemoveLayersMutation(
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
                                            new RemoveVertexFunction(spyVertexName).apply(graphBuilder)
                                    );
                            removeListener.accept(vertexToRemove);
                            return new RemoveVertexFunction(vertexToRemove).apply(graphBuilder);
                        })
                        .build()));
    }

    // Initial guess on what is good
    private ModelBuilder createSeedModelBuilder(
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
                        .setActivation(new ActivationReLU())
                        .setKernelSize(3)
                        .setNrofKernels(24))
                        .setFactory(spyFactory))
                .andThen(pool)

                .andThen(new SpyBlock(new Conv2DBatchNormAfter()
                        .setActivation(new ActivationReLU())
                        .setKernelSize(2)
                        .setNrofKernels(49))
                        .setFactory(spyFactory))
                .andThen(pool)

                .andThen(new SpyBlock(new Conv2DBatchNormAfter()
                        .setActivation(new ActivationReLU())
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
                        .setLambda(0.00299));
    }

    // Random model
    private ModelBuilder createRandomModelBuilder(
            Random rng,
            UnaryOperator<GraphBuilderAdapter> spyFactory) {

        final int schedPeriod = 50;
        final ISchedule lrSched = new Mul(new MinLim(0.02, new Step(4000, new Exponential(0.2))),
                new SawTooth(schedPeriod, 1e-6, 0.05));
        final ISchedule momSched = new Offset(schedPeriod / 2,
                new SawTooth(schedPeriod, 0.85, 0.95));

        final Function<Long, LayerBlockConfig> beforeGp = createBeforeGlobPoolLayerFactory(spyFactory, rng);
        final UnaryOperator<LayerBlockConfig> maybeResBlock = block -> rng.nextDouble() < 0.3 ? new ResBlock().setBlockConfig(block) : block;

        final int nrofPoolingLayers = rng.nextInt(2) + 1;
        final int avgLayersPerBlock = 2;
        final int avgNoutPerBlock = 16 * avgLayersPerBlock;
        final int minNout = avgNoutPerBlock / 4;
        final LayerBlockConfig pool = new Pool2D().setSize(2).setStride(2);
        final BlockBuilder.RootBuilder builder = new BlockBuilder()
                .setUpdater(new Nesterovs(lrSched, momSched))
                .setNamePrefix(namePrefix)
                .first(new ConvType(inputShape));

        for (int poolingBlockInd = 0; poolingBlockInd < nrofPoolingLayers; poolingBlockInd++) {
            final int nrofLayersThisBlock = rng.nextInt(3) + 1;
            final int thisAvgNout = avgNoutPerBlock * (poolingBlockInd + 1) / nrofLayersThisBlock;
            final long nOut = rng.nextInt(Math.max(0, thisAvgNout - minNout)) + minNout;
            for (int layerInd = 0; layerInd < nrofLayersThisBlock; layerInd++) {
                if (layerInd > 0) {
                    builder.andThen(beforeGp
                            .andThen(maybeResBlock)
                            .apply(nOut));
                } else {
                    builder.andThen(beforeGp.apply(nOut));
                }
            }
            builder.andThen(pool);
        }
        builder.andThen(new GlobPool());
        final int nrofLayersAfterGlobPool = rng.nextInt(3) + 1;
        final Function<Long, LayerBlockConfig> afterGp = createAfterGlobPoolLayerFactory(spyFactory, rng);
        int reminder = avgNoutPerBlock * (nrofPoolingLayers + 1);
        for (int layerInd = 0; layerInd < nrofLayersAfterGlobPool; layerInd++) {
            final long nOut = rng.nextInt(Math.max(0, reminder / (nrofLayersAfterGlobPool - layerInd) - minNout)) + minNout;
            reminder -= nOut;
            builder.andThen(afterGp.apply(nOut));
        }

        return builder.andFinally(new CenterLossOutput(trainIter.totalOutcomes())
                .setAlpha(0.6)
                .setLambda(0.003));
    }
}
