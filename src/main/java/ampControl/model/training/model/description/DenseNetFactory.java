package ampControl.model.training.model.description;

import ampControl.model.training.data.iterators.CachingDataSetIterator;
import ampControl.model.training.model.BlockBuilder;
import ampControl.model.training.model.GenericModelHandle;
import ampControl.model.training.model.GraphModelAdapter;
import ampControl.model.training.model.ModelHandle;
import ampControl.model.training.model.layerblocks.*;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.nd4j.linalg.learning.config.Nesterovs;

import java.nio.file.Path;
import java.util.List;
import java.util.stream.DoubleStream;

/**
 * Description of a bunch of architectures which belong to a family of dense nets.
 * <br><br>
 * See https://arxiv.org/abs/1608.06993
 *
 * @author Christian Skärby
 */
public class DenseNetFactory {
    private final CachingDataSetIterator trainIter;
    private final CachingDataSetIterator evalIter;
    private final int[] inputShape;
    private final String namePrefix;
    private final Path modelDir;

    public DenseNetFactory(CachingDataSetIterator trainIter, CachingDataSetIterator evalIter, int[] inputShape, String namePrefix, Path modelDir) {
        this.trainIter = trainIter;
        this.evalIter = evalIter;
        this.inputShape = inputShape;
        this.namePrefix = namePrefix;
        this.modelDir = modelDir;
    }

    /**
     * Adds the ModelHandles defined by this class to the given list
     * @param modelData list to add models to
     */
    public void addModelData(List<ModelHandle> modelData) {
//            final int growthRate = 12;
//            final int depth = (40 - 4) / 3;
//            final IActivation act = new ActivationReLU();
//            DoubleStream.of(0, 0.2).forEach(dropOutProb -> {
        // No idea about perf...
//                BlockBuilder bBuilder = new BlockBuilder()
//                        .setNamePrefix(namePrefix)
//                        //.setTrainWs(WorkspaceMode.SEPARATE)
//                        .first(new ConvType(inputShape))
//                        //.andThen(new ZeroPad().setPad_h(5).setPad_w(5))
//                        .andThen(new Conv2D()
//                                .setNrofKernels(2 * growthRate)
//                                .setKernelSize(5)
//                                .setActivation(new ActivationIdentity()))
//                        .andThen(new Pool2D()
//                                .setSize(4)
//                                .setStride(2))
//                        .andThenStack(2)
//                        .aggDenseStack(depth / 2)
//                        .aggOf(new Conv2DBatchNormBefore()
//                                .setKernelSize(1)
//                                .setNrofKernels(4 * growthRate)
//                        .setActivation(act))
//                        .andThen(new DropOut().setDropProb(dropOutProb))
//                        .andThen(new Conv2DBatchNormBefore()
//                                .setKernelSize(3)
//                                .setNrofKernels(growthRate)
//                                .setActivation(act))
//                        .andThen(new ZeroPad().setPad_h(1).setPad_w(1))
//                        .andFinally(new DropOut().setDropProb(dropOutProb))
//                        .andThen(new Conv2D()
//                                .setNrofKernels(depth * growthRate / 2)
//                                .setKernelSize(1)
//                                .setActivation(new ActivationIdentity()))
//                        .andThen(new DropOut().setDropProb(dropOutProb))
//                        .andFinally(new Pool2D()
//                                .setSize(2)
//                                .setStride(2)
//                                .setType(PoolingType.AVG))
//                        .andThenDenseStack(depth / 2)
//                        .aggOf(new Conv2DBatchNormBefore()
//                                .setKernelSize(1)
//                                .setNrofKernels(4 * growthRate)
//                                .setActivation(act))
//                        .andThen(new DropOut().setDropProb(dropOutProb))
//                        .andThen(new Conv2DBatchNormBefore()
//                                .setKernelSize(3)
//                                .setNrofKernels(growthRate)
//                                .setActivation(act))
//                        .andThen(new ZeroPad()
//                                .setPad_w(1)
//                                .setPad_h(1))
//                        .andFinally(new DropOut().setDropProb(dropOutProb))
//                        .andThen(new Pool2D()
//                                .setSize(8)
//                                .setType(PoolingType.AVG))
//                        // .andThen(new GlobPool())
//                        // .andThen(new Dense().setActivation(new ActivationSELU()))
//                        .andFinally(new Output(trainIter.totalOutcomes()));
//                modelData.add(new GenericModelHandle(trainIter, evalIter, new GraphModelAdapter(bBuilder.buildGraph(modelDir.toString())), bBuilder.name(), bBuilder.getAccuracy()));
//            });

        //        DoubleStream.of(0, 0.2).forEach(dropOutProb -> {
        // Works poorly
//            BlockBuilder bBuilder = new BlockBuilder()
//                    .setNamePrefix(namePrefix)
//                    .first(new ConvType(inputShape))
//                    .andThen(new ZeroPad().setPad_h(5).setPad_w(5))
//                    .andThen(new Conv2D()
//                            .setNrofKernels(32)
//                            .setKernelSize(6)
//                            .setActivation(new ActivationIdentity()))
//                    .andThen(new Pool2D()
//                            .setSize(3)
//                            .setStride(2))
//                    .andThenStack(3)
//                    .aggDenseStack(3)
//                    .aggOf(new Conv2DBatchNormBefore()
//                            .setKernelSize(3)
//                            .setNrofKernels(64)
//                            .setActivation(new ActivationReLU()))
//                    .andFinally(new ZeroPad().setPad_h(1).setPad_w(1))
//                    .andThen(new Conv2D()
//                            .setNrofKernels(32 * 3)
//                            .setKernelSize(1)
//                            .setActivation(new ActivationIdentity()))
//                    .andThen(new DropOut().setDropProb(dropOutProb))
//                    .andFinally(new Pool2D()
//                            .setSize(2)
//                            .setStride(2)
//                            .setType(PoolingType.AVG))
//                    .andThenDenseStack(3)
//                    .aggOf(new Conv2DBatchNormBefore()
//                            .setKernelSize(3)
//                            .setNrofKernels(128)
//                            .setActivation(new ActivationReLU()))
//                    .andThen(new DropOut().setDropProb(dropOutProb))
//                    .andFinally(new ZeroPad()
//                            .setPad_w(1)
//                            .setPad_h(1))
//                    // .andThen(new Pool2D().setSize(8))
//                    // .andThen(new GlobPool())
//                    .andThen(new Dense().setActivation(new ActivationSELU()))
//                    .andThen(new DropOut().setDropProb(dropOutProb))
//                    .andFinally(new Output(trainIter.totalOutcomes()));
//            modelData.add(new GenericModelHandle(trainIter, evalIter, new GraphModelAdapter(bBuilder.buildGraph(modelDir.toString())), bBuilder.name(), bBuilder.getAccuracy()));
//        });
//
        DoubleStream.of(0).forEach(dropOutProb -> {
            //Better than prev ~0.96
            BlockBuilder bBuilder = new BlockBuilder()
                    .setNamePrefix(namePrefix)
                    .setStartingLearningRate(0.005)
                    .setUpdater(new Nesterovs(0.9))
                    .first(new ConvType(inputShape))
                    .andThen(new ZeroPad().setPad_h(5).setPad_w(5))
                    .andThen(new Conv2DBatchNormAfter()
                            .setNrofKernels(32)
                            .setKernelSize(6))
                    .andThen(new Pool2D()
                            .setSize(3)
                            .setStride(2))
                    .andThenStack(3)
                    .aggDenseStack(3)
                    .aggOf(new Conv2DBatchNormAfter()
                            .setKernelSize(3)
                            .setNrofKernels(64))
                    .andFinally(new ZeroPad().setPad_h(1).setPad_w(1))
                    .andThen(new Conv2DBatchNormAfter()
                            .setNrofKernels(32 * 3)
                            .setKernelSize(1))
                    //   .andThen(new DropOut().setDropProb(dropOutProb))
                    .andFinally(new Pool2D()
                            .setSize(2)
                            .setStride(2)
                            .setType(PoolingType.AVG))
                    .andThenDenseStack(3)
                    .aggOf(new Conv2DBatchNormAfter()
                            .setKernelSize(3)
                            .setNrofKernels(128))
                    //   .andThen(new DropOut().setDropProb(dropOutProb))
                    .andFinally(new ZeroPad()
                            .setPad_w(1)
                            .setPad_h(1))
                    //  .andThen(new Pool2D().setSize(8))
                    // .andThen(new GlobPool())
                    .andThenStack(2)
                    .of(new Dense().setHiddenWidth(512))//.setActivation(new ActivationSELU()))
                    .andThen(new DropOut().setDropProb(dropOutProb))
                    .andFinally(new Output(trainIter.totalOutcomes()));
            modelData.add(new GenericModelHandle(trainIter, evalIter, new GraphModelAdapter(bBuilder.buildGraph(modelDir.toString())), bBuilder.getName(), bBuilder.getAccuracy()));
        });
//
//        BlockBuilder bBuilder = new BlockBuilder()
//                .setNamePrefix(namePrefix)
//                .first(new ConvType(inputShape))
//                .andThen(new ZeroPad().setPad_h(5).setPad_w(5))
//                .andThen(new Pool2D().setSize(2).setStride(2))
//                .andThen(new Conv2DBatchNormAfter()
//                        .setNrofKernels(64)
//                        .setKernelSize(3))
//                .andThen(new SeBlock().setOutputChannels(64))
//                .andFinally(new Output(trainIter.totalOutcomes()));
//        modelData.add(new GenericModelHandle(trainIter, evalIter, new GraphModelAdapter(bBuilder.buildGraph(modelDir.toString())), bBuilder.name(), bBuilder.getAccuracy()));


//        DoubleStream.of(0, 0.2).forEach(dropOutProb -> {
//            BlockBuilder bBuilder = new BlockBuilder()
//                    .setNamePrefix(namePrefix)
//                    .first(new ConvType(inputShape))
//                    .andThen(new ZeroPad().setPad(5))
//                    .andThen(new Conv2DBatchNormAfter()
//                            .setNrofKernels(32)
//                            .setKernelSize(6))
//                    .andThen(new SeBlock())
//                    .andThen(new Pool2D()
//                            .setSize(3)
//                            .setStride(2))
//                    .andThenStack(2)
//                    .aggDenseStack(2)
//                    .aggOf(new Conv2DBatchNormAfter()
//                            .setKernelSize(3)
//                            .setNrofKernels(64))
//                    .andThen(new SeBlock())
//                    .andFinally(new ZeroPad().setPad(1))
//                    .andThen(new Conv2DBatchNormAfter()
//                            .setNrofKernels(32 * 2)
//                            .setKernelSize(1))
//                    .andThen(new SeBlock())
//                    .andThen(new DropOut().setDropProb(dropOutProb))
//                    .andFinally(new Pool2D()
//                            .setSize(2)
//                            .setStride(2)
//                            .setType(PoolingType.AVG))
//                    .andThenDenseStack(3)
//                    .aggOf(new Conv2DBatchNormAfter()
//                            .setKernelSize(3)
//                            .setNrofKernels(128))
//                    .andThen(new SeBlock())
//                    .andThen(new DropOut().setDropProb(dropOutProb))
//                    .andFinally(new ZeroPad().setPad(1))
//
//                    //  .andThen(new Pool2D().setSize(8))
//                    // .andThen(new GlobPool())
//                    .andThen(new Dense().setActivation(new ActivationSELU()))
//                    .andThen(new DropOut().setDropProb(dropOutProb))
//                    .andFinally(new Output(trainIter.totalOutcomes()));
//            modelData.add(new GenericModelHandle(trainIter, evalIter, new GraphModelAdapter(bBuilder.buildGraph(modelDir.toString())), bBuilder.name(), bBuilder.getAccuracy()));
//        });


//        DoubleStream.of(0, 0.2).forEach(dropOutProb -> {
//            BlockBuilder bBuilder = new BlockBuilder()
//                    .setNamePrefix(namePrefix)
//                    .first(new ConvType(inputShape))
//                    .andThen(new Conv2DBatchNormAfter()
//                            .setNrofKernels(32)
//                            .setKernelSize(4))
//                    .andThen(new SeBlock())
//                    .andThen(new Pool2D()
//                            .setSize(2)
//                            .setStride(2))
//                    .andThenDenseStack(2)
//                    .aggOf(new Conv2DBatchNormAfter()
//                            .setKernelSize(3)
//                            .setNrofKernels(32))
//                    .andThen(new SeBlock())
//                    .andThen(new DropOut().setDropProb(dropOutProb))
//                    .andFinally(new ZeroPad().setPad(1))
//                    .andThen(new Pool2D()
//                            .setSize(2)
//                            .setStride(2)
//                            .setType(PoolingType.AVG))
//                    .andThenDenseStack(2)
//                    .aggOf(new Conv2DBatchNormAfter()
//                            .setKernelSize(3)
//                            .setNrofKernels(64))
//                    .andThen(new SeBlock())
//                    .andThen(new DropOut().setDropProb(dropOutProb))
//                    .andFinally(new ZeroPad().setPad(1))
//                    .andThen(new Dense().setActivation(new ActivationSELU()))
//                    .andThen(new DropOut().setDropProb(dropOutProb))
//                    .andFinally(new Output(trainIter.totalOutcomes()));
//            modelData.add(new GenericModelHandle(trainIter, evalIter, new GraphModelAdapter(bBuilder.buildGraph(modelDir.toString())), bBuilder.name(), bBuilder.getAccuracy()));
//        });
////
//            DoubleStream.of(0).forEach(dropOutProb -> {
//                BlockBuilder bBuilder = new BlockBuilder()
//                        // 0.94 and no signs of overfitting after ~20h
//                        .setNamePrefix(namePrefix)
//                        .setStartingLearningRate(0.0005)
//                        .setUpdater(new Nesterovs(0.9))
//                        .first(new ConvType(inputShape))
//                        .andThen(new Conv2DBatchNormAfter()
//                                .setNrofKernels(32)
//                                .setKernelSize(3))
//                        .andThen(new SeBlock())
//                        .andThen(new Pool2D()
//                                .setSize(3)
//                                .setStride(3))
//                        .andThenDenseStack(5)
//                        .aggOf(new Conv2DBatchNormAfter()
//                                .setKernelSize(1)
//                                .setNrofKernels(64))
//                        .andThen(new Conv2DBatchNormAfter()
//                                .setKernelSize(3)
//                                .setNrofKernels(16))
//                        .andThen(new SeBlock())
//                       // .andThen(new DropOut().setDropProb(dropOutProb))
//                        .andFinally(new ZeroPad().setPad(1))
//                        .andThenStack(2)
//                        .of(new Dense())//.setActivation(new ActivationSELU()))
//                        .andThen(new DropOut().setDropProb(dropOutProb))
//                        .andFinally(new Output(trainIter.totalOutcomes()));
//                modelData.add(new GenericModelHandle(trainIter, evalIter, new GraphModelAdapter(bBuilder.buildGraph(modelDir.toString())), bBuilder.name(), bBuilder.getAccuracy()));
//            });

//        DoubleStream.of(0, 0.2).forEach(dropOutProb -> {
//        DoubleStream.of(0.2).forEach(dropOutProb -> {
        // 0.93 and no signs of overfitting after ~20h
//            BlockBuilder bBuilder = new BlockBuilder()
//                    .setNamePrefix(namePrefix)
//                    .first(new ConvType(inputShape))
//                    .andThen(new Conv2DBatchNormAfter()
//                            .setNrofKernels(32)
//                            .setKernelSize(3))
//                    .andThen(new Pool2D()
//                            .setSize(3)
//                            .setStride(3))
//                    .andThenDenseStack(5)
//                    .aggOf(new Conv2DBatchNormAfter()
//                            .setKernelSize(1)
//                            .setNrofKernels(64))
//                    .andThen(new Conv2DBatchNormAfter()
//                            .setKernelSize(3)
//                            .setNrofKernels(16))
//                    .andThen(new DropOut().setDropProb(dropOutProb))
//                    .andFinally(new ZeroPad().setPad(1))
//                    .andThen(new Dense().setActivation(new ActivationSELU()))
//                    .andThen(new DropOut().setDropProb(dropOutProb))
//                    .andFinally(new Output(trainIter.totalOutcomes()));
//            modelData.add(new GenericModelHandle(trainIter, evalIter, new GraphModelAdapter(bBuilder.buildGraph(modelDir.toString())), bBuilder.name(), bBuilder.getAccuracy()));
//        });
//
    }
}
