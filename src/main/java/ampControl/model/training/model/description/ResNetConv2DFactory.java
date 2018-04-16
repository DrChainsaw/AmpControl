package ampControl.model.training.model.description;

import ampControl.model.training.data.iterators.CachingDataSetIterator;
import ampControl.model.training.model.BlockBuilder;
import ampControl.model.training.model.GenericModelHandle;
import ampControl.model.training.model.GraphModelAdapter;
import ampControl.model.training.model.ModelHandle;
import ampControl.model.training.model.layerblocks.*;
import ampControl.model.training.model.layerblocks.graph.MinMaxPool;
import ampControl.model.training.model.layerblocks.graph.SeBlock;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.schedule.ScheduleType;
import org.nd4j.linalg.schedule.StepSchedule;

import java.nio.file.Path;
import java.util.List;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

/**
 * Description of some homebrewed resnets with 2D convolutions
 *
 * @author Christian Skärby
 */
public class ResNetConv2DFactory {
    private final CachingDataSetIterator trainIter;
    private final CachingDataSetIterator evalIter;
    private final int[] inputShape;
    private final String namePrefix;
    private final Path modelDir;

    public ResNetConv2DFactory(CachingDataSetIterator trainIter, CachingDataSetIterator evalIter, int[] inputShape, String namePrefix, Path modelDir) {
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
        final LayerBlockConfig zeroPad3x3 = new ZeroPad().setPad(1);
//        IntStream.of(0, 2, 10, 50).forEach(resDepth -> {
//            DoubleStream.of(0).forEach(dropOutProb -> {
//                BlockBuilder bBuilder = new BlockBuilder()
//                        .setNamePrefix(namePrefix)
//                        .setStartingLearningRate(0.001)
//                        .setUpdater(new Nesterovs(0.9))
//                        //.setTrainWs(WorkspaceMode.SEPARATE)
//                        //.setEvalWs(WorkspaceMode.SEPARATE)
//                        .first(new ConvType(inputShape))
//                        .andThen(zeroPad3x3)
//                        .andThen(new Conv2DBatchNormAfter()
//                                .setKernelSize(3)
//                                .setNrofKernels(64))
//                        .andThen(new MinMaxPool().setSize(3).setStride(3))
//                        .andThen(zeroPad3x3)
//                        .andThen(new Conv2DBatchNormAfter()
//                                .setKernelSize(3)
//                                .setNrofKernels(128))
//                        .andThen(new MinMaxPool().setSize(3).setStride(3))
//                        .andThen(new SeBlock())
//                        .andThen(zeroPad3x3)
//                        .andThen(new Conv2DBatchNormAfter()
//                                .setKernelSize(3)
//                                .setNrofKernels(128))
//                        .andThen(new MinMaxPool().setSize(3).setStride(3))
//                        .andThen(new SeBlock())
//                        .andThenStack(resDepth)
//                        .res()
//                        .aggOf(zeroPad3x3)
//                        .andThen(new Conv2DBatchNormAfter()
//                                .setKernelSize(3)
//                                .setNrofKernels(256))
//                        .andThen(zeroPad3x3)
//                        .andThen(new Conv2DBatchNormAfter()
//                                .setKernelSize(3)
//                                .setNrofKernels(256))
//                        //.andThen(zeroPad3x3)
//                        .andFinally(new SeBlock())
//                        //.andFinally(new DropOut().setDropProb(dropOutProb))
//                        .andThenStack(2)
//                        .aggOf(new Dense())
//                        .andFinally(new DropOut().setDropProb(dropOutProb))
//                        .andFinally(new Output(trainIter.totalOutcomes()));
//                modelData.add(new GenericModelHandle(trainIter, evalIter, new GraphModelAdapter(bBuilder.buildGraph(modelDir.toString())), bBuilder.getName(), bBuilder.getAccuracy()));
//            });
//        });

        IntStream.of(10).forEach(resDepth -> {
            DoubleStream.of(0).forEach(dropOutProb -> {
                DoubleStream.of(0, 0.0003, 0.003, 0.03).forEach(lambda -> {
                BlockBuilder bBuilder = new BlockBuilder()
                        .setNamePrefix(namePrefix)
                        .setUpdater(new Nesterovs(new StepSchedule(ScheduleType.ITERATION, 0.001, 0.1, 40000)))
                        //.setTrainWs(WorkspaceMode.SEPARATE)
                        //.setEvalWs(WorkspaceMode.SEPARATE)
                        .first(new ConvType(inputShape))
                        .andThen(zeroPad3x3)
                        .andThen(new Conv2DBatchNormAfter()
                                .setKernelSize(3)
                                .setNrofKernels(64))
                        .andThen(new MinMaxPool().setSize(3).setStride(3))
                        .andThen(zeroPad3x3)
                        .andThen(new Conv2DBatchNormAfter()
                                .setKernelSize(3)
                                .setNrofKernels(128))
                        .andThen(new MinMaxPool().setSize(3).setStride(3))
                        .andThen(new SeBlock())
                        .andThen(zeroPad3x3)
                        .andThen(new Conv2DBatchNormAfter()
                                .setKernelSize(3)
                                .setNrofKernels(128))
                        .andThen(new MinMaxPool().setSize(3).setStride(3))
                        .andThen(new SeBlock())
                        .andThenStack(resDepth)
                        .res()
                        .aggOf(new Conv2DBatchNormAfter()
                                .setKernelSize(1)
                                .setNrofKernels(64))
                        .andThen(zeroPad3x3)
                        .andThen(new Conv2DBatchNormAfter()
                                .setKernelSize(3)
                                .setNrofKernels(128))
                        .andThen(new Conv2DBatchNormAfter()
                                .setKernelSize(1)
                                .setNrofKernels(256))
                        //.andThen(zeroPad3x3)
                        .andFinally(new SeBlock())
                        //.andFinally(new DropOut().setDropProb(dropOutProb))
                        .andThenStack(2)
                        .aggOf(new Dense())
                        .andFinally(new DropOut().setDropProb(dropOutProb))
                        .andFinally(new CenterLossOutput(trainIter.totalOutcomes())
                                .setAlpha(0.6)
                                .setLambda(lambda));
                modelData.add(new GenericModelHandle(trainIter, evalIter, new GraphModelAdapter(bBuilder.buildGraph(modelDir.toString())), bBuilder.getName(), bBuilder.getAccuracy()));
            });
        });

    }
}
