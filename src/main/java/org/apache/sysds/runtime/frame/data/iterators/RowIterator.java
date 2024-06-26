/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.sysds.runtime.frame.data.iterators;

import java.util.Iterator;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.util.UtilFunctions;

public abstract class RowIterator<T> implements Iterator<T[]> {

	protected static final Log LOG = LogFactory.getLog(RowIterator.class.getName());

	protected final FrameBlock _fb;
	protected final int[] _cols;
	protected final T[] _curRow;
	protected final int _maxPos;
	protected int _curPos = -1;

	protected RowIterator(FrameBlock fb, int rl, int ru) {
		this(fb, rl, ru, UtilFunctions.getSeqArray(1, fb.getNumColumns(), 1));
	}

	protected RowIterator(FrameBlock fb, int rl, int ru, int[] cols) {
		if(rl < 0 || ru > fb.getNumRows() || rl > ru)
			throw new DMLRuntimeException("Invalid range of iterator: " + rl + "->" + ru);

		_fb = fb;
		_curRow = createRow(cols.length);
		_cols = cols;
		_maxPos = ru;
		_curPos = rl;
	}

	@Override
	public boolean hasNext() {
		return(_curPos < _maxPos);
	}

	@Override
	public void remove() {
		throw new DMLRuntimeException("RowIterator.remove() is unsupported!");
	}

	protected abstract T[] createRow(int size);
}
