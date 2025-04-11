import { useState, useCallback } from 'react';
import { FilterOption } from '@/services/types';

export function useFilters(initialFilters: FilterOption[] = []) {
    const [selectedFilters, setSelectedFilters] = useState<FilterOption[]>(initialFilters);
    const [appliedFilters, setAppliedFilters] = useState<FilterOption[]>(initialFilters);

    const handleFilterChange = useCallback((type: string, values: string[]) => {
        setSelectedFilters(values.map(value => ({ type, value })));
    }, []);

    const handleFilterRemove = useCallback((type: string, value: string) => {
        setSelectedFilters(prev => prev.filter(f => !(f.type === type && f.value === value)));
    }, []);

    const handleFilterClear = useCallback(() => {
        setSelectedFilters([]);
    }, []);

    const handleFilterApply = useCallback(() => {
        setAppliedFilters(selectedFilters);
    }, [selectedFilters]);

    return {
        selectedFilters,
        appliedFilters,
        handleFilterChange,
        handleFilterRemove,
        handleFilterClear,
        handleFilterApply
    };
} 