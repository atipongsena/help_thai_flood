import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { RootState } from '../../store';

export type CasesFilters = {
  priority: string[];
  status: string[];
};

type CasesState = {
  filters: CasesFilters;
  selectedCaseId?: string;
  isCreateFormOpen: boolean;
  newCaseLocation?: { lat: number; lng: number };
};

const initialState: CasesState = {
  filters: { priority: [], status: [] },
  isCreateFormOpen: false,
};

const casesSlice = createSlice({
  name: 'cases',
  initialState,
  reducers: {
    changeFilters(state, action: PayloadAction<Partial<CasesFilters>>) {
      state.filters = { ...state.filters, ...action.payload };
    },
    setSelectedCase(state, action: PayloadAction<string | undefined>) {
      state.selectedCaseId = action.payload;
    },
    setCaseFormOpen(state, action: PayloadAction<boolean>) {
      state.isCreateFormOpen = action.payload;
    },
    setNewCaseLocation(state, action: PayloadAction<{ lat: number; lng: number } | undefined>) {
      state.newCaseLocation = action.payload;
    },
  },
});

export const { changeFilters, setSelectedCase, setCaseFormOpen, setNewCaseLocation } = casesSlice.actions;

export const selectFilters = (state: RootState) => state.cases.filters;
export const selectSelectedCase = (state: RootState) => state.cases.selectedCaseId;
export const selectCreateFormOpen = (state: RootState) => state.cases.isCreateFormOpen;
export const selectNewCaseLocation = (state: RootState) => state.cases.newCaseLocation;

export default casesSlice.reducer;

