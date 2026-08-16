import { configureStore } from '@reduxjs/toolkit';
import { casesApi } from '../features/cases/casesApi';
import casesReducer from '../features/cases/casesSlice';

export const store = configureStore({
  reducer: {
    [casesApi.reducerPath]: casesApi.reducer,
    cases: casesReducer,
  },
  middleware: (getDefault) => getDefault().concat(casesApi.middleware),
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;

