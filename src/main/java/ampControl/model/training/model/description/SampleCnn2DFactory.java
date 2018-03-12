package ampControl.model.training.model.description;

import ampControl.model.training.data.iterators.CachingDataSetIterator;
import ampControl.model.training.data.iterators.preprocs.CnnHeightWidthSwapInputPreprocessor;
import ampControl.model.training.model.BlockBuilder;
import ampControl.model.training.model.GenericModelHandle;
import ampControl.model.training.model.GraphModelAdapter;
import ampControl.model.training.model.ModelHandle;
import ampControl.model.training.model.layerblocks.*;
import ampControl.model.training.model.layerblocks.graph.PreprocVertex;
import org.nd4j.linalg.learning.config.Nesterovs;

import java.nio.file.Path;
import java.util.List;
import java.util.stream.DoubleStream;

/**
 * Sample CNN using Conv2D with width 1.
 * <br><br>
 * https://arxiv.org/abs/1710.10451
 *
 * @author Christian Skärby
 */
public class SampleCnn2DFactory {
    private final CachingDataSetIterator trainIter;
    private final CachingDataSetIterator evalIter;
    private final int[] inputShape;
    private final String namePrefix;
    private final Path modelDir;

    public SampleCnn2DFactory(CachingDataSetIterator trainIter, CachingDataSetIterator evalIter, int[] inputShape, String namePrefix, Path modelDir) {
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
        DoubleStream.of(0).forEach(dropOutProb -> {
            BlockBuilder bBuilder = new BlockBuilder()
                    .setNamePrefix(namePrefix)
                    .setUpdater(new Nesterovs(0.9))
                    .setStartingLearningRate(0.0005)
                    .first(new ConvType(inputShape))
                    .andThen(new PreprocVertex().setPreProcessor(new CnnHeightWidthSwapInputPreprocessor()))

                    //"Stem layer"
                    .andThen(new Conv2DBatchNormAfter()
                            .setNrofKernels(256)
                            .setKernelSize_h(3)
                            .setKernelSize_w(1)
                            .setStride_h(3))

                    // Block 1: 128 filters
//                            .andThenStack(2)
//                            .of(new Conv2DBatchNormAfter()
//                                    .setNrofKernels(128)
//                                    .setKernelSize_h(3)
//                                    .setKernelSize_w(1))
                    //                        .andFinally(new Pool1D().setSize(3).setStride(3))

                    // Block 1: 256 filters
                    .andThenStack(2)
                    .aggOf(new Conv2DBatchNormAfter()
                            .setNrofKernels(256)
                            .setKernelSize_h(3)
                            .setKernelSize_w(1))
                    .andFinally(new Pool2D()
                            .setSize_h(3)
                            .setSize_w(1)
                            .setStride_h(3)
                            .setSize_w(1))

                    .multiLevel()
                    .andThenAgg(new Conv2DBatchNormAfter()
                            .setNrofKernels(256)
                            .setKernelSize_h(3)
                            .setKernelSize_w(1))
                    .andFinally(new Pool2D()
                            .setSize_h(3)
                            .setSize_w(1)
                            .setStride_h(3)
                            .setSize_w(1))
                    .andThenAgg(new Conv2DBatchNormAfter()
                            .setNrofKernels(512)
                            .setKernelSize_h(3)
                            .setKernelSize_w(1))
                    .andFinally(new Pool2D()
                            .setSize_h(3)
                            .setSize_w(1)
                            .setStride_h(3)
                            .setSize_w(1))
                    .andThenAgg(new Conv2DBatchNormAfter()
                            .setNrofKernels(512)
                            .setKernelSize_h(3)
                            .setKernelSize_w(1))
                    .andFinally(new Pool2D()
                            .setSize_h(3)
                            .setSize_w(1)
                            .setStride_h(3)
                            .setSize_w(1))
                    //End of multilevel
                    .done()

                    //.andThen(new Dense())
                    // .andThen(new GlobMeanMax())
                    .andThenStack(2)
                    .aggOf(new Dense().setHiddenWidth(512))
                    .andFinally(new DropOut().setDropProb(dropOutProb))
                    .andFinally(new Output(trainIter.totalOutcomes()));

            modelData.add(new GenericModelHandle(
                    trainIter,
                    evalIter,
                    new GraphModelAdapter(bBuilder.buildGraph(modelDir.toString())),
                    bBuilder.getName(),
                    bBuilder.getAccuracy()));
        });

//            DoubleStream.of(0, 0.2).forEach(dropOutProb -> {
        // Same as above with residual connections
//                BlockBuilder bBuilder = new BlockBuilder()
//                        .setNamePrefix(namePrefix)
//                        //.setUpdater(new Nesterovs(0.9))
//                        .setStartingLearningRate(0.01)
//                        .first(new ConvType(inputShape))
//                        .andThen(new PreprocVertex().setPreProcessor(new CnnHeightWidthSwapInputPreprocessor()))
//
//                        //"Stem layer"
//                        .andThen(new Conv2DBatchNormAfter()
//                                .setNrofKernels(256)
//                                .setKernelSize_h(3)
//                                .setKernelSize_w(1)
//                                .setStride_h(3))
//
//
//                        // Block 1: 256 filters
//                        .andThenStack(2)
//                        .aggRes()
//                        .aggOf(new ZeroPad().setPad_h(1).setPad_w(0))
//                        .andFinally(new Conv2DBatchNormAfter()
//                                .setNrofKernels(256)
//                                .setKernelSize_h(3)
//                                .setKernelSize_w(1))
//                        .andFinally(new Pool2D()
//                                .setSize_h(3)
//                                .setSize_w(1)
//                                .setStride_h(3)
//                                .setSize_w(1))
//
//                        .multiLevel()
//                        .andThenAggRes()
//                        .aggOf(new ZeroPad().setPad_h(1).setPad_w(0))
//                        .andFinally(new Conv2DBatchNormAfter()
//                                .setNrofKernels(256)
//                                .setKernelSize_h(3)
//                                .setKernelSize_w(1))
//                        .andFinally(new Pool2D()
//                                .setSize_h(3)
//                                .setSize_w(1)
//                                .setStride_h(3)
//                                .setSize_w(1))
//                        .andThenAggRes()
//                        .aggOf(new ZeroPad().setPad_h(1).setPad_w(0))
//                        .andFinally(new Conv2DBatchNormAfter()
//                                .setNrofKernels(256)
//                                .setKernelSize_h(3)
//                                .setKernelSize_w(1))
//                        .andFinally(new Pool2D()
//                                .setSize_h(3)
//                                .setSize_w(1)
//                                .setStride_h(3)
//                                .setSize_w(1))
//                        .andThenAgg(new Conv2DBatchNormAfter()
//                                .setNrofKernels(512)
//                                .setKernelSize_h(3)
//                                .setKernelSize_w(1))
//                        .andFinally(new Pool2D()
//                                .setSize_h(3)
//                                .setSize_w(1)
//                                .setStride_h(3)
//                                .setSize_w(1))
//                        //End of multilevel
//                        .done()
//
//                        //.andThen(new Dense())
//                        // .andThen(new GlobMeanMax())
//                        .andThenStack(2)
//                        .aggOf(new Dense().setHiddenWidth(512))
//                        .andFinally(new DropOut().setDropProb(dropOutProb))
//                        .andFinally(new Output(trainIter.totalOutcomes()));
//
//                modelData.add(new GenericModelHandle(
//                        trainIter,
//                        evalIter,
//                        new GraphModelAdapter(bBuilder.buildGraph(modelDir.toString())),
//                        bBuilder.name(),
//                        bBuilder.getAccuracy()));
//            });
//
//            DoubleStream.of(0, 0.2).forEach(dropOutProb -> {
        // Same as above with SeBlocks
//                BlockBuilder bBuilder = new BlockBuilder()
//                        .setNamePrefix(namePrefix)
//                        //.setUpdater(new Nesterovs(0.9))
//                        .setStartingLearningRate(0.01)
//                        .first(new ConvType(inputShape))
//                        .andThen(new PreprocVertex().setPreProcessor(new CnnHeightWidthSwapInputPreprocessor()))
//
//                        //"Stem layer"
//                        .andThen(new Conv2DBatchNormAfter()
//                                .setNrofKernels(256)
//                                .setKernelSize_h(3)
//                                .setKernelSize_w(1)
//                                .setStride_h(3))
//                        .andThen(new SeBlock())
//
//
//                        // Block 1: 256 filters
//                        .andThenStack(2)
//                        .aggRes()
//                        .aggOf(new ZeroPad().setPad_h(1).setPad_w(0))
//                        .andThen(new Conv2DBatchNormAfter()
//                                .setNrofKernels(256)
//                                .setKernelSize_h(3)
//                                .setKernelSize_w(1))
//                        .andFinally(new SeBlock())
//                        .andFinally(new Pool2D()
//                                .setSize_h(3)
//                                .setSize_w(1)
//                                .setStride_h(3)
//                                .setSize_w(1))
//
//                        .multiLevel()
//                        .andThenAggRes()
//                        .aggOf(new ZeroPad().setPad_h(1).setPad_w(0))
//                        .andThen(new Conv2DBatchNormAfter()
//                                .setNrofKernels(256)
//                                .setKernelSize_h(3)
//                                .setKernelSize_w(1))
//                        .andFinally(new SeBlock())
//                        .andFinally(new Pool2D()
//                                .setSize_h(3)
//                                .setSize_w(1)
//                                .setStride_h(3)
//                                .setSize_w(1))
//                        .andThenAggRes()
//                        .aggOf(new ZeroPad().setPad_h(1).setPad_w(0))
//                        .andThen(new Conv2DBatchNormAfter()
//                                .setNrofKernels(256)
//                                .setKernelSize_h(3)
//                                .setKernelSize_w(1))
//                        .andFinally(new SeBlock())
//                        .andFinally(new Pool2D()
//                                .setSize_h(3)
//                                .setSize_w(1)
//                                .setStride_h(3)
//                                .setSize_w(1))
//                        .andThenAgg(new Conv2DBatchNormAfter()
//                                .setNrofKernels(512)
//                                .setKernelSize_h(3)
//                                .setKernelSize_w(1))
//                        .andThen(new SeBlock())
//                        .andFinally(new Pool2D()
//                                .setSize_h(3)
//                                .setSize_w(1)
//                                .setStride_h(3)
//                                .setSize_w(1))
//                        //End of multilevel
//                        .done()
//
//                        //.andThen(new Dense())
//                        // .andThen(new GlobMeanMax())
//                        .andThenStack(2)
//                        .aggOf(new Dense().setHiddenWidth(512))
//                        .andFinally(new DropOut().setDropProb(dropOutProb))
//                        .andFinally(new Output(trainIter.totalOutcomes()));
//
//                modelData.add(new GenericModelHandle(
//                        trainIter,
//                        evalIter,
//                        new GraphModelAdapter(bBuilder.buildGraph(modelDir.toString())),
//                        bBuilder.name(),
//                        bBuilder.getAccuracy()));
//            });

////
////            DoubleStream.of(0).forEach(dropOutProb -> {
////                BlockBuilder bBuilder = new BlockBuilder()
////                        .setNamePrefix(namePrefix)
////                        //.setUpdater(new Nesterovs(0.9))
////                        .setStartingLearningRate(0.01)
////                        .first(new ConvType(inputShape))
////                        .andThen(new PreprocVertex().setPreProcessor(new CnnHeightWidthSwapInputPreprocessor()))
////
////                        //"Stem layer"
////                        .andThen(new Conv2DBatchNormAfter()
////                                .setNrofKernels(256)
////                                .setKernelSize_h(3)
////                                .setKernelSize_w(1)
////                                .setStride_h(3))
////                        .andThen(new SeBlock().setReduction(1/16d))
////
////                        // Block 1: 128 filters
////                        //                        .andThenStack(2)
////                        //                        .aggOf(new Conv1D()
////                        //                                .setNrofKernels(128)
////                        //                                .setKernelSize(3)
////                        //                                .setStride(1))
////                        //                        .andFinally(new Pool1D().setSize(3).setStride(3))
////
////                        // Block 1: 256 filters
////                        .andThenStack(2)
////                        .aggOf(new Conv2DBatchNormAfter()
////                                .setNrofKernels(256)
////                                .setKernelSize_h(3)
////                                .setKernelSize_w(1))
////                        .andThen(new SeBlock().setReduction(1/16d))
////                        .andFinally(new Pool2D()
////                                .setSize_h(3)
////                                .setSize_w(1)
////                                .setStride_h(3)
////                                .setSize_w(1))
////
////                        .multiLevel()
////                        .andThenAgg(new Conv2DBatchNormAfter()
////                                .setNrofKernels(256)
////                                .setKernelSize_h(3)
////                                .setKernelSize_w(1))
////                        .andThen(new SeBlock().setReduction(1/16d))
////                        .andFinally(new Pool2D()
////                                .setSize_h(3)
////                                .setSize_w(1)
////                                .setStride_h(3)
////                                .setSize_w(1))
////                        .andThenAgg(new Conv2DBatchNormAfter()
////                                .setNrofKernels(256)
////                                .setKernelSize_h(3)
////                                .setKernelSize_w(1))
////                        .andThen(new SeBlock().setReduction(1/16d))
////                        .andFinally(new Pool2D()
////                                .setSize_h(3)
////                                .setSize_w(1)
////                                .setStride_h(3)
////                                .setSize_w(1))
////                        .andThenAgg(new Conv2DBatchNormAfter()
////                                .setNrofKernels(512)
////                                .setKernelSize_h(3)
////                                .setKernelSize_w(1))
////                        .andThen(new SeBlock().setReduction(1/16d))
////                        .andFinally(new Pool2D()
////                                .setSize_h(3)
////                                .setSize_w(1)
////                                .setStride_h(3)
////                                .setSize_w(1))
////                        //End of multilevel
////                        .done()
////
////                        //.andThen(new Dense())
////                        // .andThen(new GlobMeanMax())
////                        .andThenStack(2)
////                        .aggOf(new Dense())
////                        .andFinally(new DropOut().setDropProb(dropOutProb))
////                        .andFinally(new Output(trainIter.totalOutcomes()));
////
////                modelData.add(new GenericModelHandle(
////                        trainIter,
////                        evalIter,
////                        new GraphModelAdapter(bBuilder.buildGraph(modelDir.toString())),
////                        bBuilder.name(),
////                        bBuilder.getAccuracy()));
////            });


//            Conv2DBatchNormBefore conv256 = new Conv2DBatchNormBefore()
//                    .setNrofKernels(256)
//                    .setKernelSize_h(3)
//                    .setKernelSize_w(1)
//                    .setActivation(new ActivationReLU());
//            Conv2DBatchNormBefore conv256NoAct = new Conv2DBatchNormBefore()
//                    .setNrofKernels(256)
//                    .setKernelSize_h(3)
//                    .setKernelSize_w(1)
//                    .setActivation(new ActivationIdentity());
//
//            Conv2DBatchNormBefore conv512 = new Conv2DBatchNormBefore()
//                    .setNrofKernels(512)
//                    .setKernelSize_h(3)
//                    .setKernelSize_w(1)
//                    .setActivation(new ActivationReLU());
//            Conv2DBatchNormBefore conv512NoAct = new Conv2DBatchNormBefore()
//                    .setNrofKernels(512)
//                    .setKernelSize_h(3)
//                    .setKernelSize_w(1)
//                    .setActivation(new ActivationIdentity());
//            DoubleStream.of(0, 0.2).forEach(dropOutProb -> {
//                // Very heavy
//                BlockBuilder bBuilder = new BlockBuilder()
//                        .setNamePrefix(namePrefix)
//                        //.setUpdater(new Nesterovs(0.9))
//                        .setStartingLearningRate(0.01)
//                        .first(new ConvType(inputShape))
//                        .andThen(new PreprocVertex().setPreProcessor(new CnnHeightWidthSwapInputPreprocessor()))
//
//                        //"Stem layer"
//                        .andThen(new ZeroPad().setPad_h(1).setPad_w(0))
//                        .andThen(new Conv2DBatchNormBetween()
//                                .setNrofKernels(256)
//                                .setKernelSize_h(3)
//                                .setKernelSize_w(1)
//                                .setActivation(new ActivationReLU())
//                                .setStride_h(3))
//                        .andThen(new SeBlock().setReduction(1/16d))
//
//
//
//                        // Block 1: 256 filters
//                        .andThenStack(2)
//                        .aggRes()
//                        .aggOf(new ZeroPad().setPad_h(1).setPad_w(0))
//                        .andThen(conv256)
//                        .andThen(new ZeroPad().setPad_h(1).setPad_w(0))
//                        .andThen(conv256NoAct)
//                        .andFinally(new SeBlock().setReduction(1/16d))
//                        .andFinally(new Pool2D()
//                                .setSize_h(3)
//                                .setSize_w(1)
//                                .setStride_h(3)
//                                .setSize_w(1))
//
//
////                        .andThen(new ZeroPad().setPad_h(1).setPad_w(0))
////                        .andThen(new Conv2DBatchNormAfter()
////                                .setNrofKernels(256)
////                                .setKernelSize_h(1)
////                                .setKernelSize_w(1))
////                        .andThen(new SeBlock().setReduction(1/4d))
//
//                        .multiLevel()
//                        .andThenRes()
//                        .aggOf(new ZeroPad().setPad_h(1).setPad_w(0))
//                        .andThen(conv256)
//                        .andThen(new ZeroPad().setPad_h(1).setPad_w(0))
//                        .andThen(conv256NoAct)
//                        .andFinally(new SeBlock().setReduction(1/16d))
//                        .andThen(new Pool2D()
//                                .setSize_h(3)
//                                .setSize_w(1)
//                                .setStride_h(3)
//                                .setSize_w(1))
//                        .andThenRes()
//                        .aggOf(new ZeroPad().setPad_h(1).setPad_w(0))
//                        .andThen(conv256)
//                        .andThen(new ZeroPad().setPad_h(1).setPad_w(0))
//                        .andThen(conv256NoAct)
//                        .andFinally(new SeBlock().setReduction(1/16d))
//                        .andThen(new Pool2D()
//                                .setSize_h(3)
//                                .setSize_w(1)
//                                .setStride_h(3)
//                                .setSize_w(1))
//                        .andThen(new ZeroPad().setPad_h(1).setPad_w(0))
//                        .andThen(new Conv2DBatchNormAfter()
//                                .setNrofKernels(512)
//                                .setKernelSize_h(3)
//                                .setKernelSize_w(1))
//                        .andThen(new SeBlock().setReduction(1/16d))
//                        .andThenRes()
//                        .aggOf(new ZeroPad().setPad_h(1).setPad_w(0))
//                        .andThen(conv512)
//                        .andThen(new ZeroPad().setPad_h(1).setPad_w(0))
//                        .andThen(conv512NoAct)
//                        .andFinally(new SeBlock().setReduction(1/16d))
//                        .andThen(new Act())
//                        .andThen(new Pool2D()
//                                .setSize_h(3)
//                                .setSize_w(1)
//                                .setStride_h(3)
//                                .setSize_w(1))
//                        // End of multilevel
//                        .done()
//
//                        //.andThen(new GlobMeanMax())
//                        .andThenStack(2)
//                        .aggOf(new Dense().setHiddenWidth(1024))
//                        .andFinally(new DropOut().setDropProb(dropOutProb))
//                        .andFinally(new Output(trainIter.totalOutcomes()));
//
//                modelData.add(new GenericModelHandle(
//                        trainIter,
//                        evalIter,
//                        new GraphModelAdapter(bBuilder.buildGraph(modelDir.toString())),
//                        bBuilder.name(),
//                        bBuilder.getAccuracy()));
//            });
    }
}
