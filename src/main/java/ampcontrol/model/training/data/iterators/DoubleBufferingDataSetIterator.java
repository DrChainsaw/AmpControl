package ampcontrol.model.training.data.iterators;

import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.util.List;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Double buffered {@link DataSetIterator}. Idea is to load part of the data set in parallel while the model
 * is working on another part. Uses two {@link CachingDataSetIterator}s for buffering.
 *
 * @author Christian Skärby
 */
public class DoubleBufferingDataSetIterator implements DataSetIterator {

    private BufferNode current;

    private final static class BufferNode {
        private final CachingDataSetIterator iter;
        private final Lock lock;
        private BufferNode next;

        private boolean isReset = true;

        private BufferNode(DataSetIterator iter, int bufferSize, Lock lock) {
            this.iter = new CachingDataSetIterator(iter, bufferSize);
            this.lock = lock;
        }

        private void prepare() {
            iter.reset();
            iter.initCache(lock);
        }

        private void reset() {
            lock.lock();
            try {
                isReset = true;
                iter.reset();
            } finally {
                lock.unlock();
            }
        }

        private DataSet nextDataSet() {
            if(isReset) {
                lock.lock();
                try {
                    return iter.next();
                } finally {
                    isReset = false;
                    lock.unlock();
                }
            }
            return iter.next();
        }
    }

    /**
     * Constructor
     * @param sourceIter Source iterator for which output shall be buffered
     * @param bufferSize Size of buffer
     */
    public DoubleBufferingDataSetIterator(DataSetIterator sourceIter, int bufferSize) {
        final Lock lock = new ReentrantLock();
        final BufferNode first = new BufferNode(sourceIter, bufferSize, lock);
        final BufferNode second = new BufferNode(sourceIter, bufferSize, lock);
        first.next = second;
        second.next = first;
        this.current = first;
    }

    public DoubleBufferingDataSetIterator initCache() {
        current.next.prepare();
        return this;
    }

    @Override
    public DataSet next(int num) {
        throw new UnsupportedOperationException("Not supported!");
    }

    @Override
    public int inputColumns() {
        return current.iter.inputColumns();
    }

    @Override
    public int totalOutcomes() {
        return current.iter.totalOutcomes();
    }

    @Override
    public boolean resetSupported() {
        return current.iter.resetSupported() && current.next.iter.resetSupported();
    }

    @Override
    public boolean asyncSupported() {
        return current.iter.asyncSupported() && current.next.iter.asyncSupported();
    }

    @Override
    public void reset() {
        current.reset();
        current.next.prepare();
    }

    @Override
    public int batch() {
        return current.iter.batch();
    }

    @Override
    public void setPreProcessor(DataSetPreProcessor preProcessor) {
        current.iter.setPreProcessor(preProcessor);
        current.next.iter.setPreProcessor(preProcessor);
    }

    @Override
    public DataSetPreProcessor getPreProcessor() {
        return current.iter.getPreProcessor();
    }

    @Override
    public List<String> getLabels() {
        return current.iter.getLabels();
    }

    @Override
    public boolean hasNext() {
        return current.iter.hasNext() || current.next.iter.hasNext();
    }

    @Override
    public synchronized DataSet next() {

        if (!current.iter.hasNext()) {
            current.prepare();
            current = current.next;
        }

        return current.nextDataSet();
    }
}
