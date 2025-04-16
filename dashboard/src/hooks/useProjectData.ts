import { useState, useEffect } from 'react';
import { StatusCount, SubtaskTypeInfo, FilterOption } from '@/services/types';
import firebaseService from '@/services/FirebaseService';

export function useProjectData(projectId: string) {
    const [taskCounts, setTaskCounts] = useState<StatusCount[]>([]);
    const [subtaskCounts, setSubtaskCounts] = useState<StatusCount[]>([]);
    const [subtaskTypeInfos, setSubtaskTypeInfos] = useState<SubtaskTypeInfo[]>([]);
    const [error, setError] = useState<string | null>(null);
    const [isLoading, setIsLoading] = useState(false);

    useEffect(() => {
        let isMounted = true;

        const fetchData = async () => {
            if (!projectId) {
                setIsLoading(false);
                return;
            }

            try {
                setIsLoading(true);
                setError(null);

                const [taskData, subtaskData, typeData] = await Promise.all([
                    firebaseService.getTaskStatusCounts(projectId),
                    firebaseService.getSubtaskStatusCounts(projectId),
                    firebaseService.getSubtaskTypesWithCounts(projectId)
                ]);

                if (isMounted) {
                    setTaskCounts(taskData);
                    setSubtaskCounts(subtaskData);
                    setSubtaskTypeInfos(typeData);
                }
            } catch (error) {
                console.error('Error fetching project data:', error);
                if (isMounted) {
                    setError('Failed to load project data.');
                }
            } finally {
                if (isMounted) {
                    setIsLoading(false);
                }
            }
        };

        fetchData();

        return () => {
            isMounted = false;
        };
    }, [projectId]);

    const fetchFilteredData = async (filters: FilterOption[]) => {
        if (!projectId) {
            setIsLoading(false);
            return;
        }

        try {
            setIsLoading(true);
            setError(null);

            const subtaskTypes = filters
                .filter(f => f.type === 'type')
                .map(f => f.value);

            const [taskData, subtaskData] = await Promise.all([
                firebaseService.getTaskStatusCounts(projectId),
                firebaseService.getSubtaskStatusCounts(projectId, subtaskTypes)
            ]);

            setTaskCounts(taskData);
            setSubtaskCounts(subtaskData);
        } catch (error) {
            console.error('Error fetching filtered data:', error);
            setError('Failed to load filtered data.');
        } finally {
            setIsLoading(false);
        }
    };

    return {
        taskCounts,
        subtaskCounts,
        subtaskTypeInfos,
        error,
        isLoading,
        fetchFilteredData
    };
} 