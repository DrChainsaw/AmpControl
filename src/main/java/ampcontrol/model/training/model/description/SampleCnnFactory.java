package ampcontrol.model.training.model.description;

import ampcontrol.model.training.data.iterators.CachingDataSetIterator;
import ampcontrol.model.training.data.iterators.preprocs.Cnn2DtoCnn1DInputPreprocessor;
import ampcontrol.model.training.model.*;
import ampcontrol.model.training.model.layerblocks.*;
import ampcontrol.model.training.model.layerblocks.graph.PreprocVertex;
import ampcontrol.model.training.model.layerblocks.graph.ZeroPad1D;

import java.nio.file.Path;
import java.util.List;
import java.util.stream.DoubleStream;

/**
 * Sample CNN using Conv1D. Inferior performance as things like batch normalization for 1D conv does not exist in dl4j
 * 9.1. Use the 2D verions instead.
 * <br><br>
 * https://arxiv.org/abs/1710.10451
 *
 * @author Christian Skärby
 */
public class SampleCnnFactory {
    private final CachingDataSetIterator trainIter;
    private final CachingDataSetIterator evalIter;
    private final int[] inputShape;
    private final String namePrefix;
    private final Path modelDir;

    public SampleCnnFactory(CachingDataSetIterator trainIter, CachingDataSetIterator evalIter, int[] inputShape, String namePrefix, Path modelDir) {
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
        DoubleStream.of(0).forEach(dropOutProb -> {
            // .95 with potential room for improvment after 70k iters
            ModelBuilder builder = new DeserializingModelBuilder(modelDir.toString(),
                    new BlockBuilder()
                            .setNamePrefix(namePrefix)
                            //.setUpdater(new Nesterovs(0.9))
                            .setStartingLearningRate(0.01)
                            .first(new ConvTimeType(inputShape))
                            .andThen(new PreprocVertex()
                                    .setPreProcessor(new Cnn2DtoCnn1DInputPreprocessor()))

                            //"Stem layer"
                            .andThen(new Conv1D()
                                    .setNrofKernels(256)
                                    .setKernelSize(3)
                                    .setStride(3))

                            // Block 1: 128 filters
//                        .andThenStack(2)
//                        .aggOf(new Conv1D()
//                                .setNrofKernels(128)
//                                .setKernelSize(3)
//                                .setStride(1))
//                        .andFinally(new Pool1D().setSize(3).setStride(3))

                            // Block 1: 256 filters
                            .andThenStack(2)
                            .aggOf(new Conv1D()
                                    .setNrofKernels(256)
                                    .setKernelSize(3)
                                    .setStride(1))
                            .andFinally(new Pool1D().setSize(3).setStride(3))

                            .multiLevel()
                            .andThenAgg(new Conv1D()
                                    .setNrofKernels(256)
                                    .setKernelSize(3)
                                    .setStride(1))
                            //.andThen(new DropOut().setDropProb(0.2))
                            .andFinally(new Pool1D().setSize(3).setStride(3))
                            .andThenAgg(new Conv1D()
                                    .setNrofKernels(256)
                                    .setKernelSize(3)
                                    .setStride(1))
                            //.andThen(new DropOut().setDropProb(0.2))
                            .andFinally(new Pool1D().setSize(3).setStride(3))
                            .andThenAgg(new Conv1D()
                                    .setNrofKernels(512)
                                    .setKernelSize(3)
                                    .setStride(1))
                            .andFinally(new Pool1D().setSize(3).setStride(3))
                            //End of multilevel
                            .done()

                            //.andThen(new Dense())
                            // .andThen(new GlobMeanMax())
                            .andThenStack(2)
                            .aggOf(new Dense())
                            .andFinally(new DropOut().setDropProb(dropOutProb))
                            .andFinally(new Output(trainIter.totalOutcomes())));

            modelData.add(new GenericModelHandle(
                    trainIter,
                    evalIter,
                    new GraphModelAdapter(builder.buildGraph()),
                    builder.name()));
        });


        DoubleStream.of(0).forEach(dropOutProb -> {
            // Very heavy 93.5% after 70k iters with potential improvement
            BlockBuilder bBuilder = new BlockBuilder()
                    .setNamePrefix(namePrefix)
                    //.setUpdater(new Nesterovs(0.9))
                    .setStartingLearningRate(0.01)
                    .first(new ConvTimeType(inputShape))
                    .andThen(new PreprocVertex()
                            .setPreProcessor(new Cnn2DtoCnn1DInputPreprocessor()))

                    //"Stem layer"
                    .andThen(new Conv1D()
                            .setNrofKernels(128)
                            .setKernelSize(3)
                            .setStride(3))

                    .andThen(new ZeroPad1D().setPadding(1))

                    // .andThen(new Pool1D().setSize(100).setStride(50))

                    // Block 1: 128 filters
//                        .andThenStack(2)
//                        .aggRes()
//                        .ofStack(2)
//                        .aggOf(new Conv1D()
//                                .setNrofKernels(128)
//                                .setKernelSize(3)
//                                .setStride(1))
//                        .andFinally(new ZeroPad1D().setPadding(1))
//                        // End of resblocks
//                        .andFinally(new Pool1D().setSize(3).setStride(3))

                    // Block 1: 256 filters
                    .andThenStack(2)
                    .aggRes()
                    .ofStack(2)
                    .aggOf(new ZeroPad1D().setPadding(1))
                    .andFinally(new Conv1D()
                            .setNrofKernels(256)
                            .setKernelSize(3)
                            .setStride(1))
                    // End of resblocks
                    .andFinally(new Pool1D().setSize(3).setStride(3))

                    .multiLevel()
                    .andThenAggRes()
                    .ofStack(2)
                    .aggOf(new ZeroPad1D().setPadding(1))
                    // End of aggblock in res
                    .andFinally(new Conv1D()
                            .setNrofKernels(256)
                            .setKernelSize(3)
                            .setStride(1))
                    // end of res
                    .andFinally(new Pool1D().setSize(3).setStride(3))
                    .andThenAggRes()
                    .ofStack(2)
                    .aggOf(new ZeroPad1D().setPadding(1))
                    // End of aggblock in res
                    .andFinally(new Conv1D()
                            .setNrofKernels(256)
                            .setKernelSize(3)
                            .setStride(1))
                    // end of res
                    .andFinally(new Pool1D().setSize(3).setStride(3))
                    .andThenStack(2)
                    .aggOf(new ZeroPad1D().setPadding(1))
                    .andThen(new Conv1D()
                            .setNrofKernels(512)
                            .setKernelSize(3))
                    .andThenRes()
                    .ofStack(2)
                    .aggOf(new ZeroPad1D().setPadding(1)
                    )
                    // End of aggblock in res
                    .andFinally(new Conv1D()
                            .setNrofKernels(512)
                            .setKernelSize(3)
                            .setStride(1))
                    // end of res
                    .andFinally(new Pool1D().setSize(3).setStride(3))
                    // End of multilevel
                    .done()

                    //.andThen(new GlobMeanMax())
                    .andThenStack(2)
                    .aggOf(new Dense())
                    .andFinally(new DropOut().setDropProb(dropOutProb))
                    .andFinally(new Output(trainIter.totalOutcomes()));

            modelData.add(new GenericModelHandle(
                    trainIter,
                    evalIter,
                    new GraphModelAdapter(bBuilder.buildGraph()),
                    bBuilder.name()));
        });
    }
}
