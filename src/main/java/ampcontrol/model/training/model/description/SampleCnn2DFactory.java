package ampcontrol.model.training.model.description;

import ampcontrol.model.training.data.iterators.CachingDataSetIterator;
import ampcontrol.model.training.data.iterators.preprocs.CnnHeightWidthSwapInputPreprocessor;
import ampcontrol.model.training.model.*;
import ampcontrol.model.training.model.layerblocks.*;
import ampcontrol.model.training.model.layerblocks.graph.PreprocVertex;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.schedule.ScheduleType;
import org.nd4j.linalg.schedule.StepSchedule;

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
            ModelBuilder builder = new DeserializingModelBuilder(modelDir.toString(),
                    new BlockBuilder()
                    .setNamePrefix(namePrefix)
                    .setUpdater(new Nesterovs(new StepSchedule(ScheduleType.ITERATION, 0.0005, 0.1, 40000)))
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
                    .andFinally(new Output(trainIter.totalOutcomes())));

            modelData.add(new GenericModelHandle(
                    trainIter,
                    evalIter,
                    new GraphModelAdapter(builder.buildGraph()),
                    builder.name()));
        });
    }
}
