package ampControl.model.training.model.description;

import ampControl.model.training.data.iterators.CachingDataSetIterator;
import ampControl.model.training.model.BlockBuilder;
import ampControl.model.training.model.GenericModelHandle;
import ampControl.model.training.model.GraphModelAdapter;
import ampControl.model.training.model.layerblocks.*;
import ampControl.model.training.model.ModelHandle;
import ampControl.model.training.model.layerblocks.graph.MinMaxPool;
import ampControl.model.training.model.layerblocks.graph.SeBlock;
import org.nd4j.linalg.activations.impl.ActivationReLU;
import org.nd4j.linalg.learning.config.Nesterovs;

import java.nio.file.Path;
import java.util.List;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * Description of a bunch of architectures which belong to a family of stacked 2D convolutional neural networks.
 *
 * @author Christian Skärby
 */
public class StackedConv2DFactory {
    private final CachingDataSetIterator trainIter;
    private final CachingDataSetIterator evalIter;
    private final int[] inputShape;
    private final String namePrefix;
    private final Path modelDir;

    public StackedConv2DFactory(CachingDataSetIterator trainIter, CachingDataSetIterator evalIter, int[] inputShape, String namePrefix, Path modelDir) {
        this.trainIter = trainIter;
        this.evalIter = evalIter;
        this.inputShape = inputShape;
        this.namePrefix = namePrefix;
        this.modelDir = modelDir;
    }

    /**
     * Adds the ModelHandles defined by this class to the given list
     *
     * @param modelData list to add models to
     */
    public void addModelData(List<ModelHandle> modelData) {
//            final LayerBlockConfig zeroPad4x4 = new ZeroPad()
//                    .setPad_h_top(1)
//                    .setPad_h_bot(2)
//                    .setPad_w_left(1)
//                    .setPad_w_right(2);
        // Best so far with LogScale spectrogram without dropout. 94.8% but seriously flattened out progress
//            IntStream.of(3).forEach(kernelSize -> {
//                DoubleStream.of(0).forEach(dropOutProb -> {
//                    BlockBuilder bBuilder = new BlockBuilder()
//                            .setStartingLearningRate(0.005)
//                            .setUpdater(new Nesterovs(0.9))
//                            .setNamePrefix(namePrefix)
//                            //.setTrainWs(WorkspaceMode.SEPARATE)
//                            // .setEvalWs(WorkspaceMode.SEPARATE)
//                            .first(new ConvType(inputShape))
//                            // .andThen(zeroPad4x4)
//                            .andThenStack(4)
//                            .aggOf(new Conv2DBatchNormAfter()
//                                    .setKernelSize(kernelSize)
//                                    .setNrofKernels(128))
//                            //.andThen(zeroPad4x4)
//                            //.andThen(new DropOut().setDropProb(dropOutProb))
//                            .andFinally(new Pool2D().setSize(2).setStride(2))
//                            //.andThen(new GlobMeanMax())
//                            .andThenStack(2)
//                            .aggOf(new Dense()
//                                    .setHiddenWidth(512)
//                                    .setActivation(new ActivationReLU()))
//                            .andFinally(new DropOut().setDropProb(dropOutProb))
//                            .andFinally(new Output(trainIter.totalOutcomes()));
//                    modelData.add(new GenericModelHandle(trainIter, evalIter, new GraphModelAdapter(bBuilder.buildGraph(modelDir.toString())), bBuilder.name(), bBuilder.getAccuracy()));
//                });
//            });

        // Best acc (95.9%) with mfsc and uszm, but seems to perform worse then the above
        // Almost as good acc and perf with lgsc
//            IntStream.of(3).forEach(kernelSize -> {
//                DoubleStream.of(0).forEach(dropOutProb -> {
//                    BlockBuilder bBuilder = new BlockBuilder()
//                            .setStartingLearningRate(0.005)
//                            .setUpdater(new Nesterovs(0.9))
//                            .setNamePrefix(namePrefix)
//                            //.setTrainWs(WorkspaceMode.SEPARATE)
//                            // .setEvalWs(WorkspaceMode.SEPARATE)
//                            .first(new ConvType(inputShape))
//                            .andThen(new Conv2DBatchNormAfter()
//                                    .setKernelSize(kernelSize)
//                                    .setNrofKernels(128))
//                            // .andThen(zeroPad4x4)
//                            .andThenStack(4)
//                            .aggOf(new Conv2DBatchNormAfter()
//                                    .setKernelSize(kernelSize)
//                                    .setNrofKernels(128))
//                            //.andThen(zeroPad4x4)
//                            //.andThen(new DropOut().setDropProb(dropOutProb))
//                            .andFinally(new Pool2D().setSize(2).setStride(2))
//                            //.andThen(new GlobMeanMax())
//                            .andThenStack(2)
//                            .aggOf(new Dense()
//                                    .setHiddenWidth(512)
//                                    .setActivation(new ActivationReLU()))
//                            .andFinally(new DropOut().setDropProb(dropOutProb))
//                            .andFinally(new Output(trainIter.totalOutcomes()));
//                    modelData.add(new GenericModelHandle(trainIter, evalIter, new GraphModelAdapter(bBuilder.buildGraph(modelDir.toString())), bBuilder.name(), bBuilder.getAccuracy()));
//                });
//            });
//
        //More or less same perf as above but with extra pars.
//            IntStream.of(3).forEach(kernelSize -> {
//                DoubleStream.of(0).forEach(dropOutProb -> {
//                    BlockBuilder bBuilder = new BlockBuilder()
//                            .setStartingLearningRate(0.005)
//                            .setUpdater(new Nesterovs(0.9))
//                            .setNamePrefix(namePrefix)
//                            //.setTrainWs(WorkspaceMode.SEPARATE)
//                            // .setEvalWs(WorkspaceMode.SEPARATE)
//                            .first(new ConvType(inputShape))
//                            .andThen(new Conv2DBatchNormAfter()
//                                    .setKernelSize(kernelSize)
//                                    .setNrofKernels(128))
//                            // .andThen(zeroPad4x4)
//                            .andThenStack(4)
//                            .aggOf(new Conv2DBatchNormAfter()
//                                    .setKernelSize(kernelSize)
//                                    .setNrofKernels(128))
//                            //.andThen(zeroPad4x4)
//                            //.andThen(new DropOut().setDropProb(dropOutProb))
//                            .andFinally(new MinMaxPool().setSize(2).setStride(2))
//                            //.andThen(new GlobMeanMax())
//                            .andThenStack(2)
//                            .aggOf(new Dense()
//                                    .setHiddenWidth(512)
//                                    .setActivation(new ActivationReLU()))
//                            .andFinally(new DropOut().setDropProb(dropOutProb))
//                            .andFinally(new Output(trainIter.totalOutcomes()));
//                    modelData.add(new GenericModelHandle(trainIter, evalIter, new GraphModelAdapter(bBuilder.buildGraph(modelDir.toString())), bBuilder.name(), bBuilder.getAccuracy()));
//                });
//            });
//
        // Significatly worse than others around, but not useless
//            IntStream.of(3).forEach(kernelSize -> {
//                DoubleStream.of(0).forEach(dropOutProb -> {
//                    BlockBuilder bBuilder = new BlockBuilder()
//                            .setStartingLearningRate(0.005)
//                            .setUpdater(new Nesterovs(0.9))
//                            .setNamePrefix(namePrefix)
//                            .first(new ConvType(inputShape))
//                            .andThen(new Conv2DBatchNormAfter()
//                                    .setKernelSize(kernelSize)
//                                    .setNrofKernels(128))
//                            .andThenStack(4)
//                            .aggOf(new Conv2DBatchNormAfter()
//                                    .setKernelSize(kernelSize)
//                                    .setNrofKernels(128))
//                            .andFinally(new Pool2D().setSize(2).setStride(2))
//                            .andThen(new Conv2D().setKernelSize(1).setNrofKernels(256))
//                            .andFinally(new Output(trainIter.totalOutcomes()));
//                    modelData.add(new GenericModelHandle(trainIter, evalIter, new GraphModelAdapter(bBuilder.buildGraph(modelDir.toString())), bBuilder.name(), bBuilder.getAccuracy()));
//                });
//            });

        // Current best score with lgsc 96.0. Also performs very well in practice
        final LayerBlockConfig pool = new MinMaxPool().setSize(2).setStride(2);
//        IntStream.of(3).forEach(kernelSize -> {
//            DoubleStream.of(0).forEach(dropOutProb -> {
//                BlockBuilder bBuilder = new BlockBuilder()
//                        .setStartingLearningRate(0.005)
//                        .setUpdater(new Nesterovs(0.9))
//                        .setNamePrefix(namePrefix)
//                        //.setTrainWs(WorkspaceMode.SEPARATE)
//                        // .setEvalWs(WorkspaceMode.SEPARATE)
//                        .first(new ConvType(inputShape))
//                        // .andThen(zeroPad4x4)
//                        .andThen(new Conv2DBatchNormAfter()
//                                .setKernelSize(kernelSize)
//                                .setNrofKernels(64))
//                        .andThen(new Conv2DBatchNormAfter()
//                                .setKernelSize(kernelSize)
//                                .setNrofKernels(64))
//                        .andThen(pool)
//
//                        .andThen(new Conv2DBatchNormAfter()
//                                .setKernelSize(kernelSize)
//                                .setNrofKernels(128))
//                        .andThen(new Conv2DBatchNormAfter()
//                                .setKernelSize(kernelSize)
//                                .setNrofKernels(128))
//                        .andThen(pool)
//
//                        .andThen(new Conv2DBatchNormAfter()
//                                .setKernelSize(kernelSize)
//                                .setNrofKernels(256))
//                        .andThen(new Conv2DBatchNormAfter()
//                                .setKernelSize(kernelSize)
//                                .setNrofKernels(256))
//                        .andThen(pool)
//
//                        .andThen(new Conv2DBatchNormAfter()
//                                .setKernelSize(kernelSize)
//                                .setNrofKernels(512))
//                        .andThen(new Conv2DBatchNormAfter()
//                                .setKernelSize(kernelSize)
//                                .setNrofKernels(512))
//                        .andThen(pool)
//
//                        //.andThen(new GlobMeanMax())
//                        .andThenStack(2)
//                        .aggOf(new Dense()
//                                .setHiddenWidth(512)
//                                .setActivation(new ActivationReLU()))
//                        .andFinally(new DropOut().setDropProb(dropOutProb))
//                        .andFinally(new Output(trainIter.totalOutcomes()));
//                modelData.add(new GenericModelHandle(trainIter, evalIter, new GraphModelAdapter(bBuilder.buildGraph(modelDir.toString())), bBuilder.getName(), bBuilder.getAccuracy()));
//            });
//        });

        Stream.of(new IdBlock(), new SeBlock()).forEach(afterConvBlock -> {
            IntStream.of(3).forEach(kernelSize -> {
                DoubleStream.of(0).forEach(dropOutProb -> {
                    BlockBuilder bBuilder = new BlockBuilder()
                            .setStartingLearningRate(0.005)
                            .setUpdater(new Nesterovs(0.9))
                            .setNamePrefix(namePrefix)
                            //.setTrainWs(WorkspaceMode.SEPARATE)
                            // .setEvalWs(WorkspaceMode.SEPARATE)
                            .first(new ConvType(inputShape))
                            .andThenStack(2)
                            .of(new Conv2DBatchNormAfter()
                                    .setKernelSize(kernelSize)
                                    .setNrofKernels(64))
                            .andThen(afterConvBlock)
                            .andThen(pool)

                            .andThenStack(2)
                            .of(new Conv2DBatchNormAfter()
                                    .setKernelSize(kernelSize)
                                    .setNrofKernels(128))
                            .andThen(afterConvBlock)
                            .andThen(pool)

                            .andThenStack(2)
                            .of(new Conv2DBatchNormAfter()
                                    .setKernelSize(kernelSize)
                                    .setNrofKernels(256))
                            .andThen(afterConvBlock)
                            .andThen(pool)

                            .andThenStack(2)
                            .of(new Conv2DBatchNormAfter()
                                    .setKernelSize(kernelSize)
                                    .setNrofKernels(512))
                            .andThen(afterConvBlock)
                            .andThen(pool)

                            //.andThen(new GlobMeanMax())
                            .andThenStack(2)
                            .aggOf(new Dense()
                                    .setHiddenWidth(512)
                                    .setActivation(new ActivationReLU()))
                            .andFinally(new DropOut().setDropProb(dropOutProb))
                            .andFinally(new Output(trainIter.totalOutcomes()));
                    modelData.add(new GenericModelHandle(trainIter, evalIter, new GraphModelAdapter(bBuilder.buildGraph(modelDir.toString())), bBuilder.getName(), bBuilder.getAccuracy()));
                });
            });
        });

//        DoubleStream.of(0).forEach(dropOutProb -> {
//            BlockBuilder bBuilder = new BlockBuilder()
//                    .setNamePrefix(namePrefix)
//                    //.setTrainWs(WorkspaceMode.SEPARATE)
//                    // .setEvalWs(WorkspaceMode.SEPARATE)
//                    .first(new ConvType(inputShape))
//                    .andThen(zeroPad4x4)
//                    .andThenStack(4)
//                    .aggOf(new Conv2DBatchNormAfter()
//                            .setKernelSize(4)
//                            .setNrofKernels(128))
//                    .andThen(zeroPad4x4)
//                    .andThen(new SeBlock())
//                    // .andThen(new DropOut().setDropProb(dropOutProb))
//                    .andFinally(new Pool2D().setSize(2).setStride(2))
//                    // .andThen(new GlobMeanMax())
//                    .andThenStack(2)
//                    .of(new Dense().setActivation(new ActivationSELU()))
//                    .andThen(new DropOut().setDropProb(dropOutProb))
//                    .andFinally(new Output(trainIter.totalOutcomes()));
//            modelData.add(new GenericModelHandle(trainIter, evalIter, new GraphModelAdapter(bBuilder.buildGraph(modelDir.toString())), bBuilder.name(), bBuilder.getAccuracy()));
//        });
//
//        DoubleStream.of(0).forEach(dropOutProb -> {
//            BlockBuilder bBuilder = new BlockBuilder()
//                    .setNamePrefix(namePrefix)
//                    //.setTrainWs(WorkspaceMode.SEPARATE)
//                    //.setEvalWs(WorkspaceMode.SEPARATE)
//                    .first(new ConvType(inputShape))
//                    .andThen(zeroPad4x4)
//                    .andThen(new Conv2DBatchNormAfter()
//                            .setKernelSize(4)
//                            .setNrofKernels(128))
//                    .andThen(zeroPad4x4)
//                    .andThenStack(3)
//                    .aggRes()
//                    .aggOf(new Conv2DBatchNormAfter()
//                            .setKernelSize(4)
//                            .setNrofKernels(128))
//                    .andFinally(zeroPad4x4)
//                    .andFinally(new Pool2D().setSize(2).setStride(2))
//                    // .andThen(new GlobMeanMax())
//                    .andThenStack(2)
//                    .of(new Dense().setActivation(new ActivationSELU()))
//                    .andThen(new DropOut().setDropProb(dropOutProb))
//                    .andFinally(new Output(trainIter.totalOutcomes()));
//            modelData.add(new GenericModelHandle(trainIter, evalIter, new GraphModelAdapter(bBuilder.buildGraph(modelDir.toString())), bBuilder.name(), bBuilder.getAccuracy()));
//        });
//
//            LayerBlockConfig zeroPad3x3 = new ZeroPad().setPad(1);
//            DoubleStream.of(0).forEach(dropOutProb -> {
//                BlockBuilder bBuilder = new BlockBuilder()
//                        .setNamePrefix(namePrefix)
//                        .setStartingLearningRate(0.005)
//                        .setUpdater(new Nesterovs(0.9))
//                        //.setTrainWs(WorkspaceMode.SEPARATE)
//                        //.setEvalWs(WorkspaceMode.SEPARATE)
//                        .first(new ConvType(inputShape))
//                        .andThen(zeroPad3x3)
//                        .andThen(new Conv2DBatchNormAfter()
//                                .setKernelSize(3)
//                                .setActivation(new ActivationReLU())
//                                .setNrofKernels(64))
//                        .andThenStack(3)
//                        .aggRes()
//                        .aggOf(zeroPad3x3)
//                        .andThen(new Conv2DBatchNormAfter()
//                                .setKernelSize(3)
//                                .setActivation(new ActivationReLU())
//                                .setNrofKernels(64))
//                        //.andThen(zeroPad3x3)
//                        .andFinally(new SeBlock())
//                        //.andFinally(new DropOut().setDropProb(dropOutProb))
//                        .andFinally(new Pool2D().setSize(2).setStride(2))
//                         .andThen(new GlobMeanMax())
////                        .andThenStack(2)
////                        .aggOf(new Dense().setActivation(new ActivationSELU()))
////                        .andFinally(new DropOut().setDropProb(dropOutProb))
//                        .andFinally(new Output(trainIter.totalOutcomes()));
//                modelData.add(new GenericModelHandle(trainIter, evalIter, new GraphModelAdapter(bBuilder.buildGraph(modelDir.toString())), bBuilder.name(), bBuilder.getAccuracy()));
//            });
    }
}
