package ampcontrol.model.training.model.description;

import ampcontrol.model.training.data.iterators.MiniEpochDataSetIterator;
import ampcontrol.model.training.data.iterators.preprocs.Cnn2DtoCnn1DInputPreprocessor;
import ampcontrol.model.training.model.GenericModelHandle;
import ampcontrol.model.training.model.GraphModelAdapter;
import ampcontrol.model.training.model.ModelHandle;
import ampcontrol.model.training.model.builder.BlockBuilder;
import ampcontrol.model.training.model.builder.DeserializingModelBuilder;
import ampcontrol.model.training.model.builder.ModelBuilder;
import ampcontrol.model.training.model.layerblocks.*;
import ampcontrol.model.training.model.layerblocks.graph.PreprocVertex;
import ampcontrol.model.training.model.naming.FileNamePolicy;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.schedule.ScheduleType;
import org.nd4j.linalg.schedule.StepSchedule;

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
    private final MiniEpochDataSetIterator trainIter;
    private final MiniEpochDataSetIterator evalIter;
    private final int[] inputShape;
    private final String namePrefix;
    private final FileNamePolicy modelFileNamePolicy;

    public SampleCnnFactory(MiniEpochDataSetIterator trainIter, MiniEpochDataSetIterator evalIter, int[] inputShape, String namePrefix, FileNamePolicy modelFileNamePolicy) {
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
        DoubleStream.of(0).forEach(dropOutProb -> {
            // .95 with potential room for improvment after 70k iters
            ModelBuilder builder = new DeserializingModelBuilder(modelFileNamePolicy,
                    new BlockBuilder()
                            .setNamePrefix(namePrefix)
                            .setUpdater(new Adam(new StepSchedule(ScheduleType.ITERATION, 0.01, 0.1, 40000)))
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
                    .setUpdater(new Adam(new StepSchedule(ScheduleType.ITERATION, 0.01, 0.1, 40000)))
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
